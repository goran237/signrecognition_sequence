import tensorflow as tf


def generate_placeholders(pic_dim, num_labels, num_channels):
    x = tf.placeholder(tf.float32, shape=[None, pic_dim, pic_dim, num_channels])
    y_true = tf.placeholder(tf.float32, shape=[None, num_labels])
    hold_prob = tf.placeholder(tf.float32)
    return x, y_true, hold_prob
