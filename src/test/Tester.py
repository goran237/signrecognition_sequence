from model.PlaceholderGenerator import generate_placeholders
import tensorflow as tf
import numpy as np

from utils.data.process.DataHelper import DataHelper

pic_dim = 48
num_labels = 43
num_channels = 3
batch_size = 100
display_freq = 100

def perform_test():

    data_helper = DataHelper()
    #data_helper.set_up_subset_test_images_and_labels()
    data_helper.set_up_test_images_and_labels()

    with(tf.device('/cpu:0')):
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph('./models/my_test_model.meta')
            new_saver.restore(sess,tf.train.latest_checkpoint('./models/'))

            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("Input/X:0")
            y_true = graph.get_tensor_by_name("Input/Y:0")
            hold_prob = graph.get_tensor_by_name("Input/hold_prob:0")
            prediction = graph.get_tensor_by_name("Train/Prediction/predictions:0")
            accuracy = graph.get_tensor_by_name("Train/Accuracy/accuracy:0")
            loss = graph.get_tensor_by_name("Train/Loss/loss:0")

            num_test_iter = int(len(data_helper.X_test) / batch_size)
            if len(data_helper.X_test)<batch_size:
                num_test_iter = 1

            avg_loss = 0
            avg_acc = 0
            cls_pred_values = []
            tf.local_variables_initializer().run()
            for iteration in range(num_test_iter):
                batch_test = data_helper.next_batch_test(batch_size, pic_dim)
                loss_test, acc_test,cls_pred = sess.run([loss, accuracy,prediction],
                                               feed_dict={x: batch_test[0], y_true: batch_test[1], hold_prob: 1.0})
                avg_loss = avg_loss + loss_test
                avg_acc = avg_acc + acc_test
                cls_pred_values.append(cls_pred)
            print('---------------------------------------------------------')
            print('Average test loss: {0:.2f}'.format(avg_loss / num_test_iter))
            print('Average test accurracy: {0:.05%}%'.format(avg_acc / num_test_iter))
            print('---------------------------------------------------------')
            #cls_pred = sess.run(prediction, feed_dict={x:data_helper.X_test, y_true:data_helper.y_test,hold_prob:1.0})
            # for i in range(len(cls_pred_values)):
            #     correct_label = np.array(data_helper.y_test[i])
            #     if np.argmax(correct_label) == cls_pred_values[i]:
            #         pred_status = ' [O]'
            #     else:
            #         pred_status =' [X]'
            #     print('true: ' + str(np.argmax(correct_label))+'; predicted: '+ str(cls_pred_values[i])+pred_status)


