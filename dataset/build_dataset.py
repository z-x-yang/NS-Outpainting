import random
import os
from glob import glob
import numpy as np
from PIL import Image
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Model training.')
parser.add_argument('--dataset-path', type=str, default='./scenery/')
parser.add_argument('--result-path', type=str, default='./')

args = parser.parse_args()
dataset_path = args.dataset_path
result_path = args.result_path


if not os.path.exists(result_path):
    os.makedirs(result_path)

train_list = os.listdir(dataset_path)
random.shuffle(train_list)
trainset = list(map(lambda x: os.path.join(
    dataset_path, x), train_list))

testset = trainset[0:1000]
trainset = trainset[1000:]


def build_trainset(image_list, name):
    len2 = len(image_list)
    print("len=", len2)
    writer = tf.python_io.TFRecordWriter(name)
    k = 0
    for i in range(len2):

        image = Image.open(image_list[i])
        image = image.resize((432, 144), Image.BILINEAR)
        image = image.convert('RGB')

        image_bytes = image.tobytes()

        features = {}

        features['image'] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_bytes]))

        tf_features = tf.train.Features(feature=features)

        tf_example = tf.train.Example(features=tf_features)

        tf_serialized = tf_example.SerializeToString()

        writer.write(tf_serialized)
        k = k + 1
    print(k)
    writer.close()


def build_testset(image_list, name):
    len2 = len(image_list)
    print("len=", len2)
    writer = tf.python_io.TFRecordWriter(name)
    for i in range(len2):

        image = Image.open(image_list[i])
        image = image.resize((256, 128), Image.BILINEAR)
        image = image.convert('RGB')

        image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_bytes = image.tobytes()

        features = {}

        features['image'] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_bytes]))

        tf_features = tf.train.Features(feature=features)

        tf_example = tf.train.Example(features=tf_features)

        tf_serialized = tf_example.SerializeToString()

        writer.write(tf_serialized)

        # flip image
        image = image_flip

        image_bytes = image.tobytes()

        features = {}

        features['image'] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_bytes]))

        tf_features = tf.train.Features(feature=features)

        tf_example = tf.train.Example(features=tf_features)

        tf_serialized = tf_example.SerializeToString()

        writer.write(tf_serialized)

    writer.close()


print('Build testset!')
build_testset(testset, result_path + "/testset.tfr")
print('Build trainset!')
build_trainset(trainset, result_path + "/trainset.tfr")

print('Done!')
