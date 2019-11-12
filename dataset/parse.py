import tensorflow as tf


def parse_trainset(example_proto):

    dics = {}
    dics['image'] = tf.FixedLenFeature(shape=[], dtype=tf.string)

    parsed_example = tf.parse_single_example(
        serialized=example_proto, features=dics)
    image = tf.decode_raw(parsed_example['image'], out_type=tf.uint8)

    image = tf.reshape(image, shape=[72 * 2, 216 * 2, 3])

    image = tf.random_crop(image, [64 * 2, 128 * 2, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255.
    image = 2. * image - 1.

    return image


def parse_testset(example_proto):

    dics = {}
    dics['image'] = tf.FixedLenFeature(shape=[], dtype=tf.string)

    parsed_example = tf.parse_single_example(
        serialized=example_proto, features=dics)
    image = tf.decode_raw(parsed_example['image'], out_type=tf.uint8)

    image = tf.reshape(image, shape=[64 * 2, 128 * 2, 3])

    image = tf.cast(image, tf.float32) * (2. / 255) - 1.0
    
    return image

