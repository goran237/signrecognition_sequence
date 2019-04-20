import tensorflow

import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.python.keras import Model, Input
import os
import numpy as np
import pandas as pd
import ast
import numpy as np
import math
import os
import random
from keras.preprocessing.image import img_to_array as img_to_array
from keras.preprocessing.image import load_img as load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2

from imgaug import augmenters as iaa

seq = iaa.Sequential(
    [
        # iaa.Fliplr(0.5), # horizontally flip 50% of the images

        iaa.Affine(
            scale={"x": (0.7, 1.0), "y": (0.7, 1.0)},
            # translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
            rotate=(-15, 15),
            shear=(-7, 7),
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=255,  # (0, 255), # if mode is constant, use a cval between 0 and 255
            mode='constant'
        ),

        iaa.SomeOf((0, 3),
            [
                iaa.GaussianBlur((0, 0.95)),  # blur images with a sigma of 0 to 3.0
                iaa.Sharpen(alpha=(0, 0.10), lightness=(0.85, 1.25)),  # sharpen images
                #iaa.Emboss(alpha=(0, 0.10), strength=(0, 0.3)), # emboss images
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.025 * 255), per_channel=False),
                # add gaussian noise to images
                #iaa.CoarseDropout((0.05, 0.1), size_percent=(0.50, 0.70), per_channel=False),
                # iaa.Add((-25, 25), per_channel=False), # change brightness of images (by -10 to 10 of original value)
                #iaa.Multiply((0.75, 1.25), per_channel=False),
                # iaa.ContrastNormalization((0.85, 1.15), per_channel=False), # improve or worsen the contrast
                # distorsion
                #iaa.ElasticTransformation(alpha=(0.5, 2.0), sigma=0.25),
            ]
        )

    ],
    random_order=True
)

IMAGE_SIZE = 32

def main():
    train()

def load_image(image_path, size):
    img = cv2.imread(image_path)
    img = seq.augment_image(img)

    resized = cv2.resize(src=img, dsize=(32, 32))
    return img_to_array(resized / 255)

def train():
    image_size = IMAGE_SIZE
    image_input = Input(shape=(image_size, image_size, 3), name='input_layer')

    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(image_input)
    conv_1_pooled = MaxPooling2D(padding='same')(conv_1)

    conv_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv_1_pooled)
    conv_2_pooled = MaxPooling2D(padding='same')(conv_2)

    conv_flattened = Flatten()(conv_2_pooled)

    dense_layer_1 = Dense(128, activation='relu')(conv_flattened)
    dense_layer_1_dropout = Dropout(0.2)(dense_layer_1)

    dense_layer_2 = Dense(128, activation='relu')(dense_layer_1_dropout)
    dense_layer_2_dropout = Dropout(0.2)(dense_layer_2)

    output = Dense(43, activation='softmax', name='output_layer')(dense_layer_2_dropout)

    model = tf.keras.Model(inputs=image_input,outputs = [output])

    model.compile(optimizer='adam',
                  loss={'output_layer': 'categorical_crossentropy'},
                  metrics=['accuracy'])

    class SignRecognitionSequence(tf.keras.utils.Sequence):

        def __init__(self, df_path, data_path, im_size, batch_size, mode='train'):
            self.df = pd.read_csv(df_path, sep=';')
            self.im_size = im_size
            self.batch_size = batch_size
            self.mode = mode

            self.labels = self.df['labels'].tolist()
            self.labels = to_categorical(self.labels)
            self.image_list = self.df['image_name'].apply(lambda x: os.path.join(data_path, x)).tolist()
            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.image_list , self.labels, test_size = 0.10, random_state = 42)

        def __len__(self):
            return int(math.ceil(len(self.X_train) / float(self.batch_size)))

        def on_epoch_end(self):
            self.indexes = range(len(self.X_train))
            if self.mode == 'train':
                self.indexes = random.sample(self.indexes, k=len(self.indexes))

        def get_batch_labels(self, idx):
            # Fetch a batch of labels
            return self.y_train[idx * self.batch_size: (idx + 1) * self.batch_size]

        def get_batch_features(self, idx):
            batch_images = self.X_train[idx * self.batch_size: (1 + idx) * self.batch_size]
            return np.array([load_image(im, self.im_size) for im in batch_images])

        def __getitem__(self, idx):
            batch_x = self.get_batch_features(idx)
            batch_y = self.get_batch_labels(idx)
            return batch_x, batch_y

    seq = SignRecognitionSequence('./data/train/jpg/signrecognition_data_train.csv',
                                  './data/train/jpg/',
                                  im_size=IMAGE_SIZE,
                                  batch_size=32)
    X_valid = np.array([load_image(im,IMAGE_SIZE) for im in seq.X_valid])
    y_valid = seq.y_valid

    tensor_board = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('./model/model.h5', verbose=1,save_best_only=True),
        tensor_board
    ]

    print("Performing training...")

    model.fit_generator(generator=seq,
                        epochs=20,
                        use_multiprocessing=False,
                        workers=1,
                        callbacks=callbacks,
                        validation_data=(X_valid,y_valid))


    class SignRecognitionTestSet():
        def __init__(self,df_path,data_path):
            self.df = pd.read_csv(df_path,sep=';')
            self.labels = to_categorical(self.df['ClassId'].tolist())
            self.image_list = self.df['Filename'].apply(lambda x: os.path.join(data_path,x)).tolist()
            self.test_images = np.array([load_image(im,IMAGE_SIZE) for im in self.image_list])

    test_set = SignRecognitionTestSet('./data/test/jpg/GT-final_test.csv',
                                      './data/test/jpg')

    print("Performing test...")

    model.evaluate(test_set.test_images,test_set.labels)[1]

if __name__ == '__main__':
    main()
