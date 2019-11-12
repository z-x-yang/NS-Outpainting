import tensorflow as tf
import numpy as np
import math

class Loss():
    def __init__(self, cfg):
        self.cfg = cfg

    def masked_reconstruction_loss(self, gt, recon):
        loss_recon = tf.square(gt - recon)
        mask_values = np.ones((128, 128))
        for j in range(128):
            mask_values[:, j] = (1. + math.cos(math.pi * j / 127.0)) * 0.5
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 3)
        mask1 = tf.constant(1, dtype=tf.float32, shape=[1, 128, 128, 1])
        mask2 = tf.constant(mask_values, dtype=tf.float32, shape=[1, 128, 128, 1])
        mask = tf.concat([mask1, mask2], axis=2)
        loss_recon = loss_recon * mask
        loss_recon = tf.reduce_mean(loss_recon)
        return loss_recon

    def adversarial_loss(self, dis_fun, real, fake, name):
        adversarial_pos = dis_fun(real, name=name)
        adversarial_neg = dis_fun(fake, reuse=tf.AUTO_REUSE, name=name)

        loss_adv_D = - tf.reduce_mean(adversarial_pos - adversarial_neg)

        differences = fake - real
        alpha = tf.random_uniform(shape=[self.cfg.batch_size_per_gpu, 1, 1, 1])
        interpolates = real + tf.multiply(alpha, differences)
        gradients = tf.gradients(dis_fun(
            interpolates, reuse=tf.AUTO_REUSE, name=name), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(
            tf.square(gradients), [1, 2, 3]) + 1e-10)
        gradients_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        loss_adv_D += self.cfg.lambda_gp * gradients_penalty

        loss_adv_G = -tf.reduce_mean(adversarial_neg)

        return loss_adv_D, loss_adv_G

    def global_adversarial_loss(self, dis_fun, real, fake):
        return self.adversarial_loss(dis_fun, real, fake, 'DIS')

    def local_adversarial_loss(self, dis_fun, real, fake):
        return self.adversarial_loss(dis_fun, real, fake, 'DIS2')


    def global_and_local_adv_loss(self, model, gt, recon):

        left_half_gt = tf.slice(gt, [0, 0, 0, 0], [self.cfg.batch_size_per_gpu, 128, 128, 3])
        right_half_gt = tf.slice(gt, [0, 0, 128, 0], [self.cfg.batch_size_per_gpu, 128, 128, 3])
        right_half_recon = tf.slice(recon, [0, 0, 128, 0], [self.cfg.batch_size_per_gpu, 128, 128, 3])
        real = gt
        fake = tf.concat([left_half_gt, right_half_recon], axis=2)
        global_D, global_G = self.global_adversarial_loss(model.build_adversarial_global, real, fake)

        real = right_half_gt
        fake = right_half_recon
        local_D, local_G = self.local_adversarial_loss(model.build_adversarial_local, real, fake)

        loss_adv_D = global_D + local_D
        loss_adv_G = self.cfg.beta * global_G + (1 - self.cfg.beta) * local_G

        return loss_adv_G, loss_adv_D



    def average_losses(self, loss):
        tf.add_to_collection('losses', loss)

        # Assemble all of the losses for the current tower only.
        losses = tf.get_collection('losses')

        # Calculate the total loss for the current tower.
        regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(
            losses + regularization_losses, name='total_loss')

        # Compute the moving average of all individual losses and the total
        # loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)
        return total_loss

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            # Average over the 'tower' dimension.
            g, _ = grad_and_vars[0]

            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        # clip
        if self.cfg.clip_gradient:
            gradients, variables = zip(*average_grads)
            gradients = [
                None if gradient is None else tf.clip_by_average_norm(gradient, self.cfg.clip_gradient_value)
                for gradient in gradients]
            average_grads = zip(gradients, variables)
        return average_grads

    def feed_all_gpu(self, inp_dict, gpu_num, payload_per_gpu, images, params):
        for i in range(gpu_num):
            gt = params[i]
            start_pos = i * payload_per_gpu
            stop_pos = (i + 1) * payload_per_gpu
            inp_dict[gt] = images[start_pos:stop_pos]
        return inp_dict


