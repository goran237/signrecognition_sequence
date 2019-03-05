import tensorflow as tf


def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x, W, stride):
    # x --> [batch, H, W, Channels]
    # W --> [filter H, filter W, channels IN, channels OUT]
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def max_pool_2_by_2(x, ksize, stride, name):
    # x --> [batch, h, w, c]
    return tf.nn.max_pool(x, ksize=ksize, strides=stride, padding='SAME',name=name)


def convolutional_layer(input_x, shape, name):
    with tf.variable_scope(name):
        W = init_weights(shape)
        tf.summary.histogram('weight', W)
        b = init_bias([shape[3]])
        return tf.nn.relu(conv2d(input_x, W, 1) + b)


def normal_full_layer(input_layer, size, name):
    with tf.variable_scope(name):
        input_size = int(input_layer.get_shape()[1])
        W = init_weights([input_size, size])
        tf.summary.histogram('weight', W)
        b = init_bias([size])
        tf.summary.histogram('bias', b)
        return tf.matmul(input_layer, W) + b

