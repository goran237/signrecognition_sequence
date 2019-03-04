import logging

from model import ModelGenerator
from model.PlaceholderGenerator import generate_placeholders
import tensorflow as tf

from utils.data.process.DataHelper import DataHelper

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('results.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def perform_train():
    pic_dim = 48
    num_labels = 43
    num_channels = 3
    learning_rate = 0.001
    num_steps = 5000
    batch_size = 100

    data_helper = DataHelper()
    data_helper.set_up_images_and_labels()

    x, y_true, hold_prob = generate_placeholders(pic_dim=pic_dim, num_labels=num_labels, num_channels=num_channels)
    model = ModelGenerator.create_cnn_model(x, hold_prob)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=model.y_pred))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cross_entropy)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_steps):
            batch = data_helper.next_batch(batch_size,pic_dim)
            sess.run(train, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})

            if i%100 == 0:
                print('Currently on step {}'.format(i))
                print('Accuracy is:')
                logger.debug('Currently on step {}'.format(i))
                matches = tf.equal(tf.argmax(model.y_pred, 1), tf.argmax(y_true, 1))
                acc = tf.reduce_mean(tf.cast(matches, tf.float32))
                logger.debug('Accuracy: ', acc)
                print(sess.run(acc, feed_dict={x: data_helper.X_test, y_true: data_helper.y_test, hold_prob: 1.0}))
                print('\n')



