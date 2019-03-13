import csv
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
    epochs = 10
    pic_dim = 48
    num_labels = 43
    num_channels = 3
    learning_rate = 0.001
    batch_size = 100
    display_freq = 100
    logs_path = "./logs"

    data_helper = DataHelper()
    data_helper.set_up_train_images_and_labels()
    data_helper.set_up_test_images_and_labels()
    with(tf.device('/cpu: 0')):
        x, y_true, hold_prob = generate_placeholders(pic_dim=pic_dim, num_labels=num_labels, num_channels=num_channels)
        model = ModelGenerator.create_cnn_model(x, hold_prob)
        with tf.variable_scope('Train'):
            with tf.variable_scope('Loss'):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=model.y_pred),
                                      name='loss')
            tf.summary.scalar('loss', loss)
            with tf.variable_scope('Optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
            with tf.variable_scope('Accuracy'):
                correct_prediction = tf.equal(tf.argmax(model.y_pred, 1), tf.argmax(y_true, 1), name='correct_pred')
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
            tf.summary.scalar('accuracy', accuracy)
            with tf.variable_scope('Prediction'):
                cls_prediction = tf.argmax(model.y_pred, axis=1, name='predictions')

        saver = tf.train.Saver(max_to_keep=4)
        init = tf.global_variables_initializer()
        merged = tf.summary.merge_all()
        loss_file_obj = open('loss.csv','a')
        loss_file_obj.truncate()
        loss_file_obj.close()
        print('Training in progress...')

        with tf.Session() as sess:
            sess.run(init)
            global_step = 0
            summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
            num_tr_iter = int(len(data_helper.X_train) / batch_size)
            for epoch in range(epochs):
                print('Training epoch: {}'.format(epoch + 1))
                for iteration in range(num_tr_iter):
                    global_step += 1
                    batch = data_helper.next_batch(batch_size, pic_dim)
                    sess.run(optimizer, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})
                    if iteration % display_freq == 0:
                        loss_batch, acc_batch, summary_tr = sess.run([loss, accuracy, merged],
                                                                     feed_dict={x: batch[0], y_true: batch[1],
                                                                                hold_prob: 0.5})
                        summary_writer.add_summary(summary_tr, global_step)
                        print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.05%}".
                              format(iteration, loss_batch, acc_batch))
                    if iteration % 10 == 0:
                        with open('loss.csv', mode='a',newline='') as loss_file:
                            loss_writer = csv.writer(loss_file,delimiter=';',quotechar ='"',quoting=csv.QUOTE_MINIMAL)
                            loss_writer.writerow([epoch,iteration,loss_batch])
                feed_dict_valid = {x: data_helper.X_valid, y_true: data_helper.y_valid, hold_prob: 1.0}
                loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
                print('---------------------------------------------------------')
                print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.05%}".
                      format(epoch + 1, loss_valid, acc_valid))
                print('---------------------------------------------------------')
                saver.save(sess, 'models/my_test_model')
        print('Training completed.')




