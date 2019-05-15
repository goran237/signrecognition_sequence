import math
import os
import datetime
import random
import zipfile

import cv2
import numpy as np
import pandas as pd
import tensorflow
import tensorflow as tf
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras.preprocessing.image import img_to_array as img_to_array
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.optimizers import rmsprop
from tensorflow.python.keras.applications import VGG16

#from src.SignRecognitionSequence import SignRecognitionSequence
from src.utils.Cleaner import clean_logs
from src.utils.DataExtractor import extractData
from src.utils.ImageAugment import load_image
import matplotlib.pyplot as plt

IMAGE_SIZE = 64

def main():
    extractData()
    clean_logs()
    trained_model = train()
    test(trained_model)

def createModel():
    image_size = IMAGE_SIZE

    image_input = Input(shape=(image_size, image_size, 3), name='input_layer')

    conv_1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(image_input)
    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv_1)
    conv_1_pooled = MaxPooling2D(padding='same')(conv_1)

    conv_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv_1_pooled)
    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv_2)
    conv_2_pooled = MaxPooling2D(padding='same')(conv_2)

    conv_3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv_2_pooled)
    conv_3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv_3)
    conv_3_pooled = MaxPooling2D(padding='same')(conv_3)

    conv_4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv_3_pooled)
    conv_4 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(conv_4)
    conv_4_pooled = MaxPooling2D(padding='same')(conv_4)

    conv_flattened = Flatten()(conv_4_pooled)

    dense_layer_1 = Dense(256, activation='relu')(conv_flattened)
    dense_layer_1_dropout = Dropout(0.2)(dense_layer_1)

    # dense_layer_2 = Dense(256, activation='relu')(dense_layer_1_dropout)
    # dense_layer_2_dropout = Dropout(0.35)(dense_layer_2)

    output = Dense(43, activation='softmax', name='output_layer')(dense_layer_1_dropout)

    model = tf.keras.Model(inputs=image_input, outputs=[output])

    model.compile(optimizer=Adam(1e-4),
                  loss={'output_layer': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    return model

def train():
    from src.sequence.SignRecognitionSequence import SignRecognitionSequence
    seq =  SignRecognitionSequence('./data/train/ppm/signrecognition_data_train.csv',
                                  './data/train/ppm/',
                                  im_size=IMAGE_SIZE,
                                  batch_size=64)

    X_valid = np.array([load_image(im) for im in seq.X_valid])
    y_valid = seq.y_valid

    tensor_board = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.0001)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.02, patience=3, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

    callbacks = [
        #tf.keras.callbacks.ModelCheckpoint('models/my_model.h5', verbose=1,save_best_only=True),
        tensor_board,
        reduce_lr,
        early_stopping
    ]

    print('--------------------------')
    print("Performing training...")

    # learning_rates = [1e-3, 1e-4, 1e-5]
    # for lr in learning_rates:
    #     root_dir = os.getcwd()+'\experiments\learning_rate'
    #     dir_name = str(lr)+'_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #     currentdir = os.path.join(root_dir,dir_name)
    #     os.makedirs(currentdir)
    #     model = createModel()
    #     history = model.fit_generator(generator=seq,
    #                     epochs=3,
    #                     use_multiprocessing=False,
    #                     workers=1,
    #                     callbacks=callbacks,
    #                     validation_data=(X_valid,y_valid))
    #     loss = history.history['loss']
    #     val_loss = history.history['val_loss']
    #     accuracy = history.history['accuracy']
    #
    #     epochs = range(1, len(loss)+1)
    #     plt.plot(epochs, loss, 'bo', label='Training loss')
    #     plt.plot(epochs, val_loss, 'g', label='Validation loss')
    #     plt.plot(epochs, accuracy, 'r', label='Accuracy')
    #     plt.title('Training and validation loss')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss/Accurracy')
    #     plt.legend()
    #     plt.savefig('{0}/lr_acc_plot.png'.format(currentdir))
    #
    #     test(model)
    model = createModel()
    model.fit_generator(generator=seq,
                        epochs=20,
                        use_multiprocessing=False,
                        workers=1,
                        callbacks=callbacks,
                        validation_data=(X_valid,y_valid))
    return model


def test(model):
    class SignRecognitionTestSet():
        def __init__(self,df_path,data_path):
            self.df = pd.read_csv(df_path,sep=';')
            self.labels = to_categorical(self.df['ClassId'].tolist())
            self.image_list = self.df['Filename'].apply(lambda x: os.path.join(data_path,x)).tolist()
            self.test_images = np.array([load_image(im) for im in self.image_list])

    test_set = SignRecognitionTestSet('./data/test/ppm/GT-final_test.csv',
                                      './data/test/ppm')
    print('--------------------------')
    print("Performing test...")

    model.evaluate(test_set.test_images,test_set.labels, verbose=1)[1]

if __name__ == '__main__':
    main()

