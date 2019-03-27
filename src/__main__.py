from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras import Model

import pandas as pd
import glob
import pickle

import datetime

def main():

    with open('filenames_train', 'rb') as f:
        filenames_train_raw = pickle.load(f)

    with open('labels_train', 'rb') as f:
        labels_train_raw = pickle.load(f)

    filenames_train = tf.constant(filenames_train_raw)
    labels_train = tf.constant(labels_train_raw)
    dataset_train = tf.data.Dataset.from_tensor_slices((filenames_train, labels_train))

    def _parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        image /= 255
        return image, label

    train_images = dataset_train.map(_parse_function)

    with open('filenames_test', 'rb') as f:
        filenames_test_raw = pickle.load(f)

    with open('labels_test', 'rb') as f:
        labels_test_raw = pickle.load(f)

    filenames_test = tf.constant(filenames_test_raw)
    labels_test = tf.constant(labels_test_raw)
    dataset_test = tf.data.Dataset.from_tensor_slices((filenames_test, labels_test))
    test_images = dataset_test.map(_parse_function)

    epochs = 100
    pic_dim = 48
    num_labels = 43
    num_channels = 3
    learning_rate = 0.001
    batch_size = 100
    display_freq = 100

    train_images = train_images.shuffle(39206).batch(batch_size)
    test_images = test_images.shuffle(12360).batch(batch_size)

    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = Conv2D(48, (4, 4), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))
            self.pool1 = MaxPooling2D()
            self.conv2 = Conv2D(64, (4, 4), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))
            self.pool2 = MaxPooling2D()
            self.flatten = Flatten()
            self.d1 = Dense(1024, kernel_regularizer=tf.keras.regularizers.l1(0.01), activation='relu')
            self.drop = Dropout(0.1)
            self.batchNorm = BatchNormalization()
            self.d2 = Dense(43, activation='softmax')

        def call(self, x):
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.flatten(x)
            x = self.drop(x)
            x = self.d1(x)
            x = self.batchNorm(x)
            return self.d2(x)

    model = MyModel()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(image, label):
        with tf.GradientTape() as tape:
            predictions = model(image)
            loss = loss_object(label, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(label, predictions)

    @tf.function
    def test_step(image, label):
        predictions = model(image)
        t_loss = loss_object(label, predictions)

        test_loss(t_loss)
        test_accuracy(label, predictions)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for epoch in range(epochs):
        for image, label in train_images:
            train_step(image, label)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        for test_image, test_label in test_images:
            test_step(test_image, test_label)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))

if __name__ == '__main__':
    main()
