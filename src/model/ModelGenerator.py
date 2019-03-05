from model.HelperFunctions import *
from model.Model import Model

def create_cnn_model(input_placeholder, hold_prob_placeholder):
    img_shape = (48, 48, 1)
    current_img_dim = img_shape[0]
    num_channel = 3
    num_labels = 43

    layer_1_filter_shape = (4, 4)
    layer_1_num_of_out_feats = 48
    layer_1_pool_factor= 2

    layer_2_filter_shape = (4, 4)
    layer_2_num_of_out_feats = 64
    layer_2_pool_factor = 2

    fully_connected_neurons = 1024

    model = Model()

    #layer 1
    model.convo_1 = convolutional_layer(input_placeholder, shape=[layer_1_filter_shape[0], layer_1_filter_shape[1], num_channel, layer_1_num_of_out_feats],name='conv1')
    model.convo_1_pooling = max_pool_2_by_2(model.convo_1,ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], name='pool1')
    current_img_dim = int(current_img_dim/layer_1_pool_factor)

    #layer 2
    model.convo_2 = convolutional_layer(model.convo_1_pooling, shape=[layer_2_filter_shape[0], layer_2_filter_shape[1], layer_1_num_of_out_feats, layer_2_num_of_out_feats], name='conv2')
    model.convo_2_pooling = max_pool_2_by_2(model.convo_2, ksize=[1, 2, 2, 1],stride=[1, 2, 2, 1], name='pool2')
    current_img_dim = int(current_img_dim / layer_2_pool_factor)

    #flatten
    model.convo_2_flat = tf.reshape(model.convo_2_pooling, [-1, current_img_dim*current_img_dim*layer_2_num_of_out_feats])

    #full
    model.full_layer_one = tf.nn.relu(normal_full_layer(model.convo_2_flat, fully_connected_neurons,name='fully_connected'))

    #dropout
    model.full_one_dropout = tf.nn.dropout(model.full_layer_one, keep_prob=hold_prob_placeholder,name='full_one_dropout')

    #normal full layer from dropout
    model.y_pred = normal_full_layer(model.full_one_dropout, num_labels,name='normal_full_output')

    return model

