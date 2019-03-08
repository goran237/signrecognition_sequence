from model.PlaceholderGenerator import generate_placeholders
import tensorflow as tf
import numpy as np

from utils.data.process.DataHelper import DataHelper

pic_dim = 48
num_labels = 43
num_channels = 3

def perform_test():

    data_helper = DataHelper()
    data_helper.set_up_subset_test_images_and_labels()

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('./models/my_test_model.meta')
        new_saver.restore(sess,tf.train.latest_checkpoint('./models/'))

        graph = tf.get_default_graph()
        #all_nodes = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        x = graph.get_tensor_by_name("Input/X:0")
        y_true = graph.get_tensor_by_name("Input/Y:0")
        hold_prob = graph.get_tensor_by_name("Input/hold_prob:0")
        prediction = graph.get_tensor_by_name("Train/Prediction/predictions:0")

        tf.local_variables_initializer().run()
        cls_pred = sess.run(prediction, feed_dict={x:data_helper.X_test, y_true:data_helper.y_test,hold_prob:1.0})
        pred_status = ''
        for i in range(len(cls_pred)):
            correct_label = np.array(data_helper.y_test[i])
            if np.argmax(correct_label) == cls_pred[i]:
                pred_status = ' [O]'
            else:
                pred_status =' [X]'
            print('true: ' + str(np.argmax(correct_label))+'; predicted: '+ str(cls_pred[i])+pred_status)

