import tensorflow as tf
import tensorflow.contrib.layers as ly


class Model():
    def __init__(self, cfg):
        self.cfg = cfg

    def new_atrous_conv_layer(self, bottom, filter_shape, rate, name=None):
        with tf.variable_scope(name):
            regularizer = tf.contrib.layers.l2_regularizer(self.cfg.weight_decay)
            initializer = tf.contrib.layers.xavier_initializer()
            W = tf.get_variable(
                "W",
                shape=filter_shape,
                regularizer=regularizer,
                initializer=initializer)

            x = tf.nn.atrous_conv2d(
                bottom, W, rate, padding='SAME')
        return x

    def identity_block(self, X_input, kernel_size, filters, stage, block, is_relu=False):

        if is_relu:
            activation_fn=tf.nn.relu
            
        else:
            activation_fn=self.leaky_relu

        normalizer_fn = ly.instance_norm


        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'

        with tf.variable_scope("id_block_stage" + str(stage) + block):
            filter1, filter2, filter3 = filters
            X_shortcut = X_input
            regularizer = tf.contrib.layers.l2_regularizer(self.cfg.weight_decay)
            initializer = tf.contrib.layers.xavier_initializer()

            # First component of main path
            x = tf.layers.conv2d(X_input, filter1,
                                 kernel_size=(1, 1), strides=(1, 1), name=conv_name_base + '2a', kernel_regularizer=regularizer, kernel_initializer=initializer, use_bias=False)
            x = normalizer_fn(x)
            x = activation_fn(x)

            # Second component of main path
            x = tf.layers.conv2d(x, filter2, (kernel_size, kernel_size),
                                 padding='same', name=conv_name_base + '2b', kernel_regularizer=regularizer, kernel_initializer=initializer, use_bias=False)
            x = normalizer_fn(x)
            x = activation_fn(x)

            # Third component of main path
            x = tf.layers.conv2d(x, filter3, kernel_size=(
                1, 1), name=conv_name_base + '2c', kernel_regularizer=regularizer, kernel_initializer=initializer, use_bias=False)
            x = normalizer_fn(x)

            # Final step: Add shortcut value to main path, and pass it through
            x = tf.add(x, X_shortcut)
            x = activation_fn(x)

        return x

    def convolutional_block(self, X_input, kernel_size, filters, stage, block, stride=2, is_relu=False):
        
        if is_relu:
            activation_fn=tf.nn.relu
            
        else:
            activation_fn=self.leaky_relu

        normalizer_fn = ly.instance_norm

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'

        with tf.variable_scope("conv_block_stage" + str(stage) + block):

            regularizer = tf.contrib.layers.l2_regularizer(self.cfg.weight_decay)
            initializer = tf.contrib.layers.xavier_initializer()
            # initializer = tf.variance_scaling_initializer(scale=1.0,mode='fan_in')

            # Retrieve Filters
            filter1, filter2, filter3 = filters

            # Save the input value
            X_shortcut = X_input

            # First component of main path
            x = tf.layers.conv2d(X_input, filter1,
                                 kernel_size=(1, 1),
                                 strides=(1, 1),
                                 name=conv_name_base + '2a', kernel_regularizer=regularizer, kernel_initializer=initializer, use_bias=False)
            x = normalizer_fn(x)
            x = activation_fn(x)

            # Second component of main path
            x = tf.layers.conv2d(x, filter2, (kernel_size, kernel_size), strides=(stride, stride), name=conv_name_base +
                                 '2b', padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer, use_bias=False)
            x = normalizer_fn(x)
            x = activation_fn(x)

            # Third component of main path
            x = tf.layers.conv2d(x, filter3, (1, 1), name=conv_name_base + '2c',
                                 kernel_regularizer=regularizer, kernel_initializer=initializer, use_bias=False)
            x = normalizer_fn(x)


            # SHORTCUT PATH
            X_shortcut = tf.layers.conv2d(X_shortcut, filter3, (1, 1),
                                          strides=(stride, stride), name=conv_name_base + '1', kernel_regularizer=regularizer, kernel_initializer=initializer, use_bias=False)
            X_shortcut = normalizer_fn(X_shortcut)

            # Final step: Add shortcut value to main path, and pass it through
            # a RELU activation
            x = tf.add(X_shortcut, x)
            x = activation_fn(x)

        return x

    def leaky_relu(self, x, name=None, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    def in_lrelu(self, x, name=None):
        x = tf.contrib.layers.instance_norm(x)
        x = self.leaky_relu(x)
        return x

    def in_relu(self, x, name=None):
        x = tf.contrib.layers.instance_norm(x)
        x = tf.nn.relu(x)
        return x

    def rct(self, x):
        regularizer = tf.contrib.layers.l2_regularizer(self.cfg.weight_decay)
        output_size = x.get_shape().as_list()[3]
        size = 512
        layer_num = 2
        activation_fn = tf.tanh
        x = ly.conv2d(x, size, 1, stride=1, activation_fn=None,
                      normalizer_fn=None, padding='SAME', weights_regularizer=regularizer, biases_initializer=None)
        x = self.in_lrelu(x)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [-1, 4, 4 * size])
        x = tf.transpose(x, [1, 0, 2])
        # encoder_inputs = x
        x = tf.reshape(x, [-1, 4 * size])
        x_split = tf.split(x, 4, 0)

        ys = []
        with tf.variable_scope('LSTM'):
            with tf.variable_scope('encoder'):
                lstm_cell = tf.contrib.rnn.LSTMCell(
                    4 * size, activation=activation_fn)
                lstm_cell = tf.contrib.rnn.MultiRNNCell(
                    [lstm_cell] * layer_num, state_is_tuple=True)
            
            init_state = lstm_cell.zero_state(self.cfg.batch_size_per_gpu, dtype=tf.float32)
            now, _state = lstm_cell(x_split[0], init_state)
            now, _state = lstm_cell(x_split[1], _state)
            now, _state = lstm_cell(x_split[2], _state)
            now, _state = lstm_cell(x_split[3], _state)

            with tf.variable_scope('decoder'):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                    4 * size, activation=activation_fn)
                lstm_cell2 = tf.contrib.rnn.MultiRNNCell(
                    [lstm_cell] * layer_num, state_is_tuple=True)
            #predict
            now, _state = lstm_cell2(x_split[3], _state)
            ys.append(tf.reshape(now, [-1, 4, 1, size]))
            now, _state = lstm_cell2(now, _state)
            ys.append(tf.reshape(now, [-1, 4, 1, size]))
            now, _state = lstm_cell2(now, _state)
            ys.append(tf.reshape(now, [-1, 4, 1, size]))
            now, _state = lstm_cell2(now, _state)
            ys.append(tf.reshape(now, [-1, 4, 1, size]))
        

        y = tf.concat(ys, axis=2)

        y = ly.conv2d(y, output_size, 1, stride=1, activation_fn=None,
                      normalizer_fn=None, padding='SAME', weights_regularizer=regularizer, biases_initializer=None)
        y = self.in_lrelu(y)
        return y


    
    def shc(self, x, shortcut, channels):
        regularizer = tf.contrib.layers.l2_regularizer(self.cfg.weight_decay)
        x = ly.conv2d(x, channels / 2, 1, stride=1, activation_fn=tf.nn.relu,
                      normalizer_fn=tf.contrib.layers.instance_norm, padding='SAME', weights_regularizer=regularizer)
        x = ly.conv2d(x, channels / 2, 3, stride=1, activation_fn=tf.nn.relu,
                      normalizer_fn=tf.contrib.layers.instance_norm, padding='SAME', weights_regularizer=regularizer)
        x = ly.conv2d(x, channels, 1, stride=1, activation_fn=None,
                      normalizer_fn=tf.contrib.layers.instance_norm, padding='SAME', weights_regularizer=regularizer)
        return tf.add(shortcut, x)


    def grb(self, x, filters, rate, name):
        activation_fn = tf.nn.relu
        normalizer_fn = ly.instance_norm
        shortcut = x
        x1 = self.new_atrous_conv_layer(x, [3, 1, filters, filters], rate, name+'_a1')
        x1 = normalizer_fn(x1)
        x1 = activation_fn(x1)
        x1 = self.new_atrous_conv_layer(x1, [1, 7, filters, filters], rate, name+'_a2')
        x1 = normalizer_fn(x1)

        x2 = self.new_atrous_conv_layer(x, [1, 7, filters, filters], rate, name+'_b1')
        x2 = normalizer_fn(x2)
        x2 = activation_fn(x2)
        x2 = self.new_atrous_conv_layer(x2, [3, 1, filters, filters], rate, name+'_b2')
        x2 = normalizer_fn(x2)

        x = tf.add(shortcut, x1)
        x = tf.add(x, x2)
        x = activation_fn(x)
        return x

    def build_reconstruction(self, images, reuse=None):

        with tf.variable_scope('GEN', reuse=reuse):
            x = images
            normalizer_fn = ly.instance_norm
            regularizer = tf.contrib.layers.l2_regularizer(self.cfg.weight_decay)
            initializer = tf.contrib.layers.xavier_initializer()
            # stage 1

            x = tf.layers.conv2d(x, filters=64, kernel_size=(4, 4), strides=(
                2, 2), name='conv0', kernel_regularizer=regularizer, padding='same', kernel_initializer=initializer, use_bias=False)
            x = self.in_lrelu(x)
            short_cut0 = x
            x = tf.layers.conv2d(x, filters=128, kernel_size=(4, 4), strides=(
                2, 2), name='conv1', padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer, use_bias=False)
            x = self.in_lrelu(x)
            short_cut1 = x

            # stage 2
            x = self.convolutional_block(x, kernel_size=3, filters=[
                                         64, 64, 256], stage=2, block='a', stride=2)
            x = self.identity_block(
                x, 3, [64, 64, 256], stage=2, block='b')
            x = self.identity_block(
                x, 3, [64, 64, 256], stage=2, block='c')
            short_cut2 = x

            # stage 3
            x = self.convolutional_block(x, kernel_size=3, filters=[128, 128, 512],
                                         stage=3, block='a', stride=2)
            x = self.identity_block(
                x, 3, [128, 128, 512], stage=3, block='b')
            x = self.identity_block(
                x, 3, [128, 128, 512], stage=3, block='c')
            x = self.identity_block(
                x, 3, [128, 128, 512], stage=3, block='d',)
            short_cut3 = x

            # stage 4
            x = self.convolutional_block(x, kernel_size=3, filters=[
                                         256, 256, 1024], stage=4, block='a', stride=2)
            x = self.identity_block(
                x, 3, [256, 256, 1024], stage=4, block='b')
            x = self.identity_block(
                x, 3, [256, 256, 1024], stage=4, block='c')
            x = self.identity_block(
                x, 3, [256, 256, 1024], stage=4, block='d')
            x = self.identity_block(
                x, 3, [256, 256, 1024], stage=4, block='e')
            short_cut4 = x
            
            # rct transfer
            train = self.rct(x)


            # stage -4
            train = tf.concat([short_cut4, train], axis=2)

            train = self.grb(train, 1024, 1, 't4')
            train = self.identity_block(
                train, 3, [256, 256, 1024], stage=-4, block='b', is_relu=True)
            train = self.identity_block(
                train, 3, [256, 256, 1024], stage=-4, block='c', is_relu=True)
            

            train = ly.conv2d_transpose(train, 512, 4, stride=2,
                                        activation_fn=None, normalizer_fn=normalizer_fn, padding='SAME', weights_initializer=initializer, weights_regularizer=regularizer, biases_initializer=None)
            sc, kp = tf.split(train, 2, axis=2)
            sc = tf.nn.relu(sc)
            merge = tf.concat([short_cut3, sc], axis=3)
            merge = self.shc(merge, short_cut3, 512)
            merge = self.in_relu(merge)
            train = tf.concat(
                [merge, kp], axis=2)


            # stage -3
            train = self.grb(train, 512, 2, 't3')
            train = self.identity_block(
                train, 3, [128, 128, 512], stage=-3, block='b', is_relu=True)
            train = self.identity_block(
                train, 3, [128, 128, 512], stage=-3, block='c', is_relu=True)
            train = self.identity_block(
                train, 3, [128, 128, 512], stage=-3, block='d', is_relu=True)
            
            

            train = ly.conv2d_transpose(train, 256, 4, stride=2,
                                        activation_fn=None, normalizer_fn=normalizer_fn, padding='SAME', weights_initializer=initializer, weights_regularizer=regularizer, biases_initializer=None)
            sc, kp = tf.split(train, 2, axis=2)
            sc = tf.nn.relu(sc)
            merge = tf.concat([short_cut2, sc], axis=3)
            merge = self.shc(merge, short_cut2, 256)
            merge = self.in_relu(merge)
            train = tf.concat(
                [merge, kp], axis=2)

            # stage -2
            train = self.grb(train, 256, 4, 't2')
            train = self.identity_block(
                train, 3, [64, 64, 256], stage=-2, block='b', is_relu=True)
            train = self.identity_block(
                train, 3, [64, 64, 256], stage=-2, block='c', is_relu=True)
            train = self.identity_block(
                train, 3, [64, 64, 256], stage=-2, block='d', is_relu=True)
            train = self.identity_block(
                train, 3, [64, 64, 256], stage=-2, block='e', is_relu=True)

            train = ly.conv2d_transpose(train, 128, 4, stride=2,
                                        activation_fn=None, normalizer_fn=normalizer_fn, padding='SAME', weights_initializer=initializer, weights_regularizer=regularizer, biases_initializer=None)
            sc, kp = tf.split(train, 2, axis=2)
            sc = tf.nn.relu(sc)
            merge = tf.concat([short_cut1, sc], axis=3)
            merge = self.shc(merge, short_cut1, 128)
            merge = self.in_relu(merge)
            train = tf.concat(
                [merge, kp], axis=2)
 

            # stage -1

            train = ly.conv2d_transpose(train, 64, 4, stride=2,
                                        activation_fn=None, normalizer_fn=normalizer_fn, padding='SAME', weights_initializer=initializer, weights_regularizer=regularizer, biases_initializer=None)
            sc, kp = tf.split(train, 2, axis=2)
            sc = tf.nn.relu(sc)
            merge = tf.concat([short_cut0, sc], axis=3)
            merge = self.shc(merge, short_cut0, 64)
            merge = self.in_relu(merge)
            train = tf.concat(
                [merge, kp], axis=2)

            # stage -0
            recon = ly.conv2d_transpose(train, 3, 4, stride=2,
                                        activation_fn=None, padding='SAME', weights_initializer=initializer, weights_regularizer=regularizer, biases_initializer=None)

        return recon, tf.nn.tanh(recon)

    def build_adversarial_global(self, img, reuse=None, name=None):
        bs = img.get_shape().as_list()[0]
        with tf.variable_scope(name, reuse=reuse):

            def lrelu(x, leak=0.2, name="lrelu"):
                with tf.variable_scope(name):
                    f1 = 0.5 * (1 + leak)
                    f2 = 0.5 * (1 - leak)
                    return f1 * x + f2 * abs(x)

            size = 128
            normalizer_fn = ly.instance_norm
            activation_fn = lrelu

            img = ly.conv2d(img, num_outputs=size / 2, kernel_size=4,
                            stride=2, activation_fn=activation_fn)
            img = ly.conv2d(img, num_outputs=size, kernel_size=4,
                            stride=2, activation_fn=activation_fn, normalizer_fn=normalizer_fn)
            img = ly.conv2d(img, num_outputs=size * 2, kernel_size=4,
                            stride=2, activation_fn=activation_fn, normalizer_fn=normalizer_fn)
            img = ly.conv2d(img, num_outputs=size * 4, kernel_size=4,
                            stride=2, activation_fn=activation_fn, normalizer_fn=normalizer_fn)
            img = ly.conv2d(img, num_outputs=size * 4, kernel_size=4,
                            stride=2, activation_fn=activation_fn, normalizer_fn=normalizer_fn)

            logit = ly.fully_connected(tf.reshape(
                img, [bs, -1]), 1, activation_fn=None)

        return logit

    def build_adversarial_local(self, img, reuse=None, name=None):
        bs = img.get_shape().as_list()[0]
        with tf.variable_scope(name, reuse=reuse):

            def lrelu(x, leak=0.2, name="lrelu"):
                with tf.variable_scope(name):
                    f1 = 0.5 * (1 + leak)
                    f2 = 0.5 * (1 - leak)
                    return f1 * x + f2 * abs(x)

            size = 128
            normalizer_fn = ly.instance_norm
            activation_fn = lrelu

            img = ly.conv2d(img, num_outputs=size / 2, kernel_size=4,
                            stride=2, activation_fn=activation_fn)
            img = ly.conv2d(img, num_outputs=size, kernel_size=4,
                            stride=2, activation_fn=activation_fn, normalizer_fn=normalizer_fn)
            img = ly.conv2d(img, num_outputs=size * 2, kernel_size=4,
                            stride=2, activation_fn=activation_fn, normalizer_fn=normalizer_fn)
            img = ly.conv2d(img, num_outputs=size * 2, kernel_size=4,
                            stride=2, activation_fn=activation_fn, normalizer_fn=normalizer_fn)

            logit = ly.fully_connected(tf.reshape(
                img, [bs, -1]), 1, activation_fn=None)

        return logit


