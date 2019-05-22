import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras.optimizers import rmsprop
from tensorflow.python.keras.utils import to_categorical

# from src.SignRecognitionSequence import SignRecognitionSequence
from src.utils.Cleaner import clean_logs
from src.utils.DataExtractor import extractData
from src.utils.ImageAugment import load_image

IMAGE_SIZE = 64

def main():
    extractData()
    clean_logs()
    trained_model = train()
    test(trained_model)

def createModel():
    image_size = IMAGE_SIZE

    image_input = Input(shape=(image_size, image_size, 3), name='input_layer')

    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(image_input)
    #conv_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv_1)
    conv_1_pooled = MaxPooling2D(padding='same')(conv_1)

    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv_1_pooled)
    #conv_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv_2)
    conv_2_pooled = MaxPooling2D(padding='same')(conv_2)

    conv_3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv_2_pooled)
    #conv_3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv_3)
    conv_3_pooled = MaxPooling2D(padding='same')(conv_3)

    conv_4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv_3_pooled)
    #conv_4 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(conv_4)
    conv_4_pooled = MaxPooling2D(padding='same')(conv_4)

    #conv_5 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv_4_pooled)
    #conv_4 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(conv_4)
    #conv_5_pooled = MaxPooling2D(padding='same')(conv_5)

    #conv_6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv_5_pooled)
    #conv_4 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(conv_4)
    #conv_6_pooled = MaxPooling2D(padding='same')(conv_6)

    conv_flattened = Flatten()(conv_4_pooled)

    dense_layer_1 = Dense(512, activation='relu')(conv_flattened)
    dense_layer_1_dropout = Dropout(0.2)(dense_layer_1)

    # dense_layer_2 = Dense(256, activation='relu')(dense_layer_1_dropout)
    # dense_layer_2_dropout = Dropout(0.35)(dense_layer_2)

    output = Dense(43, activation='softmax', name='output_layer')(dense_layer_1_dropout)

    model = tf.keras.Model(inputs=image_input, outputs=[output])

    model.compile(optimizer=rmsprop(1e-3),
                  loss={'output_layer': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    model.summary()
    return model

def train():
    from src.sequence.SignRecognitionSequence import SignRecognitionSequence
    seq =  SignRecognitionSequence('./data/train/ppm/signrecognition_data_train.csv',
                                  './data/train/ppm/',
                                  im_size=IMAGE_SIZE,
                                  batch_size=32)

    X_valid = np.array([load_image(im) for im in seq.X_valid])
    y_valid = seq.y_valid

    tensor_board = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.000001)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.02, patience=5, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

    callbacks = [
        #tf.keras.callbacks.ModelCheckpoint('models/my_model.h5', verbose=1,save_best_only=True),
        tensor_board,
        reduce_lr
    ]

    print('--------------------------')
    print("Performing training...")
    model = createModel()
    model.fit_generator(generator=seq,
                        epochs=25,
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

