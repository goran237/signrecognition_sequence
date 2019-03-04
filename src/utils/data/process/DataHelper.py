import logging

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('log_file.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def one_hot_encode(vector, vals):
    n = len(vector)
    out = np.zeros((n, vals))
    out[range(n), vector] = 1
    return out


class DataHelper:

    def __init__(self):
        self.i = 0
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

    def set_up_images_and_labels(self):
        root = './utils/data/GTSRB/Final_Training/Images'
        all_images = []
        all_oh_labels = []
        for i in range(43):
            logger.info('Processing sign: ' + str(i))
            folder_name = format(i, '05d')
            folder_name_resized = folder_name + '_R'
            csv_file_path = root + '/' + folder_name + '/' + 'GT-' + folder_name + '.csv'
            csv_info = pd.read_csv(csv_file_path, delimiter=';')
            for pic_name in csv_info['Filename']:
                current_image_path = root + '/' + folder_name_resized + '/' + pic_name
                np_image = np.array(Image.open(current_image_path))
                norm_image = np.array((np_image - np.min(np_image)) / (np.max(np_image) - np.min(np_image)))
                all_images.append(norm_image)
            for label in csv_info['ClassId']:
                all_oh_labels.append(label)
        all_oh_labels = one_hot_encode(all_oh_labels, 43)
        X_train, X_test, y_train, y_test = train_test_split(all_images, all_oh_labels, test_size= 0.1, random_state=42)
        permutation_train = np.random.permutation(len(X_train))
        permutation_test = np.random.permutation(len(X_test))
        self.X_train = np.array([X_train[idx] for idx in permutation_train], dtype=np.float32)
        self.y_train = [y_train[idx] for idx in permutation_train]
        self.X_test = np.array([X_test[idx] for idx in permutation_test], dtype=np.float32)
        self.y_test = [y_test[idx] for idx in permutation_test]
        logger.info('All images have been processed')


    def next_batch(self, batch_size,pic_dim):
        x = self.X_train[self.i: self.i+batch_size].reshape(batch_size, pic_dim, pic_dim, 3)
        y = self.y_train[self.i: self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.X_train)
        return x, y

    # TODO: make the batches method
