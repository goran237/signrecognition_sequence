import tensorflow as tf


def generate_placeholders(pic_dim, num_labels, num_channels):
    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, shape=[None, pic_dim, pic_dim, num_channels], name='X')
        y_true = tf.placeholder(tf.float32, shape=[None, num_labels], name='Y')
        hold_prob = tf.placeholder(tf.float32,name='hold_prob')
        return x, y_true, hold_prob
