import random
import os
from glob import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average
import tensorflow.contrib.layers as ly
from modeling.model import Model
from modeling.loss import Loss
from dataset.parse import parse_trainset, parse_testset
import argparse
import math

parser = argparse.ArgumentParser(description='Model testing.')
# experiment
parser.add_argument('--date', type=str, default='0817')
parser.add_argument('--exp-index', type=int, default=2)
parser.add_argument('--f', action='store_true', default=False)

# gpu
parser.add_argument('--start-gpu', type=int, default=0)
parser.add_argument('--num-gpu', type=int, default=1)

# dataset
parser.add_argument('--trainset-path', type=str, default='./dataset/trainset.tfr')
parser.add_argument('--testset-path', type=str, default='./dataset/testset.tfr')
parser.add_argument('--trainset-length', type=int, default=5041)
parser.add_argument('--testset-length', type=int, default=2000)  # we flip every image in testset

# training
parser.add_argument('--base-lr', type=float, default=0.0001)
parser.add_argument('--batch-size', type=int, default=20)
parser.add_argument('--weight-decay', type=float, default=0.00002)
parser.add_argument('--epoch', type=int, default=1500)
parser.add_argument('--lr-decay-epoch', type=int, default=1000)
parser.add_argument('--critic-steps', type=int, default=3)
parser.add_argument('--warmup-steps', type=int, default=1000)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--clip-gradient', action='store_true', default=False)
parser.add_argument('--clip-gradient-value', type=float, default=0.1)


# modeling
parser.add_argument('--beta', type=float, default=0.9)
parser.add_argument('--lambda-gp', type=float, default=10)
parser.add_argument('--lambda-rec', type=float, default=0.998)

# checkpoint
parser.add_argument('--log-path', type=str, default='./logs/')
parser.add_argument('--checkpoint-path', type=str, default=None)
parser.add_argument('--resume-step', type=int, default=0)


args = parser.parse_args()


# prepare path
base_path = args.log_path
exp_date = args.date
if exp_date is None:
    print('Exp date error!')
    import sys
    sys.exit()
exp_name = exp_date + '/' + str(args.exp_index)
print("Start Exp:", exp_name)
output_path = base_path + exp_name + '/'
model_path = output_path + 'models/'
tensorboard_path = output_path + 'log/'
result_path = output_path + 'results/'

if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
if not os.path.exists(result_path):
    os.makedirs(result_path)
elif not args.f:
    if args.checkpoint_path is None:
        print('Exp exist!')
        import sys
        sys.exit()
else:
    import shutil
    shutil.rmtree(model_path)
    os.makedirs(model_path)
    shutil.rmtree(tensorboard_path)
    os.makedirs(tensorboard_path)

# prepare gpu
num_gpu = args.num_gpu
start_gpu = args.start_gpu
gpu_id = str(start_gpu)
for i in range(num_gpu - 1):
    gpu_id = gpu_id + ',' + str(start_gpu + i + 1)
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
args.batch_size_per_gpu = int(args.batch_size / args.num_gpu)




model = Model(args)
loss = Loss(args)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

print("Start building model...")
with tf.Session(config=config) as sess:
    with tf.device('/cpu:0'):
        learning_rate = tf.placeholder(tf.float32, [])
        lambda_rec = tf.placeholder(tf.float32, [])

        train_op_G = tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=0.5, beta2=0.9)
        train_op_D = tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=0.5, beta2=0.9)


        trainset = tf.data.TFRecordDataset(filenames=[args.trainset_path])
        trainset = trainset.shuffle(args.trainset_length)
        trainset = trainset.map(parse_trainset, num_parallel_calls=args.workers)
        trainset = trainset.batch(args.batch_size).repeat()

        train_iterator = trainset.make_one_shot_iterator()
        train_im = train_iterator.get_next()

        testset = tf.data.TFRecordDataset(filenames=[args.testset_path])
        testset = testset.map(parse_testset, num_parallel_calls=args.workers)
        testset = testset.batch(args.batch_size).repeat()

        test_iterator = testset.make_one_shot_iterator()
        test_im = test_iterator.get_next()

        print('build model on gpu tower')
        models = []
        params = []
        for gpu_id in range(num_gpu):
            with tf.device('/gpu:%d' % gpu_id):
                print('tower_%d' % gpu_id)
                with tf.name_scope('tower_%d' % gpu_id):
                    with tf.variable_scope('cpu_variables', reuse=gpu_id > 0):

                        groundtruth = tf.placeholder(
                            tf.float32, [args.batch_size_per_gpu, 128, 256, 3], name='groundtruth')
                        left_gt = tf.slice(groundtruth, [0, 0, 0, 0], [args.batch_size_per_gpu, 128, 128, 3])


                        reconstruction_ori, reconstruction = model.build_reconstruction(left_gt)
                        right_recon = tf.slice(reconstruction, [0, 0, 128, 0], [args.batch_size_per_gpu, 128, 128, 3])

                        loss_rec = loss.masked_reconstruction_loss(groundtruth, reconstruction)
                        loss_adv_G, loss_adv_D = loss.global_and_local_adv_loss(model, groundtruth, reconstruction)

                        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                        loss_G = loss_adv_G * (1 - lambda_rec) + loss_rec * lambda_rec + sum(reg_losses)
                        loss_D = loss_adv_D

                        var_G = list(filter(lambda x: x.name.startswith(
                            'cpu_variables/GEN'), tf.trainable_variables()))
                        var_D = list(filter(lambda x: x.name.startswith(
                            'cpu_variables/DIS'), tf.trainable_variables()))


                        grad_g = train_op_G.compute_gradients(
                            loss_G, var_list=var_G)
                        grad_d = train_op_D.compute_gradients(
                            loss_D, var_list=var_D)

                        models.append((reconstruction, right_recon))
                        params.append(groundtruth)

        print('Done.')

        print('Start reducing towers on cpu...')

        reconstructions, right_recons = zip(*models)
        groundtruths = params
        
        with tf.device('/gpu:0'):

            reconstructions = tf.concat(reconstructions, axis=0)
            right_recons = tf.concat(right_recons, axis=0)

        print('Done.')


        iters = 0
        saver = tf.train.Saver(max_to_keep=5)
        if args.checkpoint_path is None:
            sess.run(tf.global_variables_initializer())
        else:
            print('Start loading checkpoint...')
            saver.restore(sess, args.checkpoint_path)
            iters = args.resume_step
            print('Done.')

        


        print('run eval...')


        stitch_mask1 = np.ones((args.batch_size, 128, 128, 3))
        for i in range(128):
            stitch_mask1[:, :, i, :] = 1. / 127. * (127. - i)
        stitch_mask2 = stitch_mask1[:, :, ::-1, :]


        ii = 0

        for _ in range(math.floor(args.testset_length / args.batch_size)):
            test_oris = sess.run([test_im])[0]
            origins1 = test_oris.copy()

            oris = None
            # oris
            print('oris ' + str(ii))
            for _ in range(4):
                inp_dict = {}
                inp_dict = loss.feed_all_gpu(inp_dict, args.num_gpu, args.batch_size_per_gpu, test_oris, params)

                if oris is None:
                    reconstruction_vals, prediction_vals = sess.run(
                        [reconstructions, right_recons],
                        feed_dict=inp_dict)

                    oris = reconstruction_vals
                    pred1 = oris[:, :, :128, :]
                    pred2 = oris[:, :, -128:, :]
                    gt = origins1[:, :, :128, :]
                    p1_m0 = np.concatenate((gt, pred2), axis=2)
                    p1_m1 = np.concatenate((gt * stitch_mask1 + pred1 * stitch_mask2, pred2), axis=2)
                else:
                    reconstruction_vals, prediction_vals = sess.run(
                        [reconstruction, right_recons],
                        feed_dict=inp_dict)
                    A = oris[:, :, -128:, :]
                    B = reconstruction_vals[:, :, :128, :]
                    C = A * stitch_mask1 + B * stitch_mask2 
                    oris = np.concatenate((oris[:, :, :-128, :], C, prediction_vals), axis=2)
                test_oris = np.concatenate((prediction_vals, prediction_vals), axis=2)
            predictions1 = oris

            jj = ii
            for ori, m0, m1, endless in zip(origins1, p1_m0, p1_m1, predictions1):
                name = str(jj) + '.jpg'
                ori = (255. * (ori + 1) / 2.).astype(np.uint8)
                Image.fromarray(ori).save(os.path.join(
                    result_path, 'ori_' + name))

                m0 = (255. * (m0 + 1) / 2.).astype(np.uint8)
                Image.fromarray(m0).save(os.path.join(
                    result_path, 'm0_' + name))

                m1 = (255. * (m1 + 1) / 2.).astype(np.uint8)
                Image.fromarray(m1).save(os.path.join(
                    result_path, 'm1_' + name))

                endless = (255. * (endless + 1) / 2.).astype(np.uint8)
                Image.fromarray(endless).save(os.path.join(
                    result_path, 'endless_' + name))
                jj += 1


            ii += args.batch_size
