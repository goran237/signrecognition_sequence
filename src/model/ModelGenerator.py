from model.HelperFunctions import *
from model.Model import Model

def create_cnn_model(input_placeholder, hold_prob_placeholder):
    img_shape = (48, 48, 1)
    num_channel = 3
    num_labels = 43

    kernel_shape_1 = (4, 4)
    layer_1_num_of_out_feats = 48
    layer_1_pool_factor= 2

    kernel_shape_2 = (4, 4)
    layer_2_num_of_out_feats = 64
    layer_2_pool_factor = 2
    layer_2_img_dim = int(img_shape[0] / (layer_2_pool_factor*layer_1_pool_factor))

    model = Model()
    model.convo_1 = convolutional_layer(input_placeholder, shape=[kernel_shape_1[0], kernel_shape_1[1],
                                                                  num_channel, layer_1_num_of_out_feats])
    model.convo_1_pooling = max_pool_2_by_2(model.convo_1)
    model.convo_2 = convolutional_layer(model.convo_1_pooling, shape=[kernel_shape_2[0], kernel_shape_2[1],
                                                                      layer_1_num_of_out_feats, layer_2_num_of_out_feats])
    model.convo_2_pooling = max_pool_2_by_2(model.convo_2)
    model.convo_2_flat = tf.reshape(model.convo_2_pooling, [-1, layer_2_img_dim*layer_2_img_dim*layer_2_num_of_out_feats])
    model.full_layer_one = tf.nn.relu(normal_full_layer(model.convo_2_flat, 1024))
    model.full_one_dropout = tf.nn.dropout(model.full_layer_one, keep_prob=hold_prob_placeholder)
    model.y_pred = normal_full_layer(model.full_one_dropout, num_labels)
    return model

