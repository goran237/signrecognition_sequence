import os

import numpy as np
import pandas as pd
import tensorflow as tf
#import tqdm
import matplotlib.pyplot as plt
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation
from tensorflow.python.keras.optimizers import rmsprop
from tensorflow.python.keras.utils import to_categorical
# from src.SignRecognitionSequence import SignRecognitionSequence
from tensorflow.python.layers.normalization import BatchNormalization

from src.utils.Cleaner import clean_logs
from src.utils.DataExtractor import extractData
from src.utils.ImageAugment import load_image
from src.utils.model_evaluation import evaluate_model
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint

from src.utils.clr_callback import CyclicLR

IMAGE_SIZE = 64
BATCH_SIZE = 32

def main():
    #extractData()
    #clean_logs()
    #trained_model = train()
    trained_model = load_model('./model/weights.sign_rec-1.10-0.0031-0.0407.h5')
    test(trained_model)


def createModel():
    image_size = IMAGE_SIZE

    image_input = Input(shape=(image_size, image_size, 3), name='input_layer')

    conv_1 = Conv2D(filters=64, kernel_size=(3, 3), use_bias=False)(image_input)
    conv_1_normalized = BatchNormalization()(conv_1)
    conv_1_activation = Activation('relu')(conv_1_normalized)
    conv_1_pooled = MaxPooling2D(padding='same')(conv_1_activation)

    conv_2 = Conv2D(filters=128, kernel_size=(3, 3), use_bias=False)(conv_1_pooled)
    conv_2_normalized = BatchNormalization()(conv_2)
    conv_2_activation = Activation('relu')(conv_2_normalized)
    conv_2_pooled = MaxPooling2D(padding='same')(conv_2_activation)

    conv_3 = Conv2D(filters=128, kernel_size=(3, 3), use_bias=False)(conv_2_pooled)
    conv_3_normalized = BatchNormalization()(conv_3)
    conv_3_activation = Activation('relu')(conv_3_normalized)
    conv_3_pooled = MaxPooling2D(padding='same')(conv_3_activation)

    conv_4 = Conv2D(filters=256, kernel_size=(3, 3), use_bias=False)(conv_3_pooled)
    conv_4_normalized = BatchNormalization()(conv_4)
    conv_4_activation = Activation('relu')(conv_4_normalized)
    conv_4_pooled = MaxPooling2D(padding='same')(conv_4_activation)

    conv_5 = Conv2D(filters=512, kernel_size=(3, 3), use_bias=False)(conv_4_pooled)
    conv_5_normalized = BatchNormalization()(conv_5)
    conv_5_activation = Activation('relu')(conv_5_normalized)
    conv_5_pooled = MaxPooling2D(padding='same')(conv_5_activation)

    conv_flattened = Flatten()(conv_5_pooled)

    dense_layer_1 = Dense(512, use_bias=False)(conv_flattened)
    dense_normalized = BatchNormalization()(dense_layer_1)
    dense_activation = Activation('relu')(dense_normalized)

    output = Dense(43, activation='softmax', name='output_layer')(dense_activation)

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
                                  batch_size=BATCH_SIZE)

    X_valid = np.array([load_image(im) for im in seq.X_valid])
    y_valid = seq.y_valid

    tensor_board = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

    #clr_triangular = CyclicLR(mode='triangular', base_lr=4e-5, max_lr=3e-4, step_size=200)
    save = ModelCheckpoint('./model/weights.sign_rec-1.{epoch:02d}-{loss:.4f}-{val_loss:.4f}.h5', period=10, verbose=1)
    callbacks = [
        #tf.keras.callbacks.ModelCheckpoint('models/my_model.h5', verbose=1,save_best_only=True),
        tensor_board,
        reduce_lr,
        #clr_triangular,
        save,
        early_stopping
    ]

    print('--------------------------')
    print("Performing training...")
    model = createModel()
    model.fit_generator(generator=seq,
                        epochs=40,
                        use_multiprocessing=False,
                        workers=1,
                        callbacks=callbacks,
                        validation_data=(X_valid,y_valid))
    return model

def test(model):
    class SignRecognitionTestSet():
        def __init__(self,df_path,data_path):
            self.df = pd.read_csv(df_path,sep=';')
            self.label_names = self.df.ClassId.unique()
            self.label_names.sort()
            self.labels = to_categorical(self.df['ClassId'].tolist())
            self.image_list = self.df['Filename'].apply(lambda x: os.path.join(data_path,x)).tolist()
            self.test_images = np.array([load_image(im) for im in self.image_list])

    test_set = SignRecognitionTestSet('./data/test/ppm/GT-final_test.csv',
                                      './data/test/ppm')
    print('--------------------------')
    print("Performing test...")

    model.evaluate(test_set.test_images,test_set.labels, verbose=1)[1]


    evaluate_model(test_set.test_images, test_set.labels, test_set.label_names, model, batch_size=32)


if __name__ == '__main__':
    main()

