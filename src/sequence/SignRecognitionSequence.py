import math
import os
import random

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import pandas as pd

from src.main import load_image


class SignRecognitionSequence(tf.keras.utils.Sequence):

    def __init__(self, df_path, data_path, im_size, batch_size, mode='train'):
        self.df = pd.read_csv(df_path, sep=';')
        self.im_size = im_size
        self.batch_size = batch_size
        self.mode = mode

        self.labels = self.df['labels'].tolist()
        self.labels = to_categorical(self.labels)
        self.image_list = self.df['image_name'].apply(lambda x: os.path.join(data_path, x)).tolist()
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.image_list, self.labels,
                                                                                  test_size=0.20, random_state=42, stratify=self.labels)

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
        return np.array([load_image(im) for im in batch_images])

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y