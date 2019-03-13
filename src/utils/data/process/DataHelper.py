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
        self.i_test = 0
        self.X_train = []
        self.y_train = []
        self.X_valid = []
        self.y_valid = []
        self.X_test = []
        self.y_test = []

    def set_up_train_images_and_labels(self):
        root_training = './utils/data/GTSRB/Final_Training/Images'
        all_images = []
        all_one_hot_labels = []
        print('Creating training datasets...')
        for i in range(43):
            logger.info('Processing training sign: ' + str(i))
            folder_name = format(i, '05d')
            folder_name_resized = folder_name + '_R'
            csv_file_path = root_training + '/' + folder_name + '/' + 'GT-' + folder_name + '.csv'
            csv_info = pd.read_csv(csv_file_path, delimiter=';')
            for pic_name in csv_info['Filename']:
                current_image_path = root_training + '/' + folder_name_resized + '/' + pic_name
                np_image = np.array(Image.open(current_image_path))
                norm_image = np.array((np_image - np.min(np_image)) / (np.max(np_image) - np.min(np_image)))
                all_images.append(norm_image)
            for label in csv_info['ClassId']:
                all_one_hot_labels.append(label)
        all_one_hot_labels = one_hot_encode(all_one_hot_labels, 43)
        X_train, X_valid, y_train, y_valid = train_test_split(all_images, all_one_hot_labels, test_size=0.1,
                                                              random_state=42)
        permutation_train = np.random.permutation(len(X_train))
        permutation_test = np.random.permutation(len(X_valid))
        self.X_train = np.array([X_train[idx] for idx in permutation_train], dtype=np.float32)
        self.y_train = [y_train[idx] for idx in permutation_train]
        self.X_valid = np.array([X_valid[idx] for idx in permutation_test], dtype=np.float32)
        self.y_valid = [y_valid[idx] for idx in permutation_test]
        logger.info('All training images have been processed')
        print('Datasets for training created.')

    def set_up_test_images_and_labels(self):
        root_test = './utils/data/GTSRB/Final_Test/Images'
        all_test_images = []
        all_test_one_hot_labels = []
        print('Creating test datasets...')
        logger.info('Processing training signs... ')
        folder_name_resized = root_test + '/resized/'
        csv_file_path = root_test + '/GT-final_test.csv'
        csv_info = pd.read_csv(csv_file_path, delimiter=';')
        for pic_name in csv_info['Filename']:
            current_image_path = root_test+'/resized/'+pic_name
            np_image = np.array(Image.open(current_image_path))
            all_test_images.append(np_image)
        for label in csv_info['ClassId']:
            all_test_one_hot_labels.append(label)
        all_test_one_hot_labels = one_hot_encode(all_test_one_hot_labels,43)
        self.X_test = all_test_images
        self.y_test = all_test_one_hot_labels
        logger.info('All test images have been processed')
        print('Datasets for test created.')

    def set_up_subset_test_images_and_labels(self):
        root_test = './utils/data/GTSRB/Final_Test/subset'
        all_test_images = []
        all_test_one_hot_labels = []
        print('Creating test datasets...')
        logger.info('Processing training signs... ')
        folder_name_resized = root_test + '/resized/'
        csv_file_path = root_test + '/GT-final_test_subset.csv'
        csv_info = pd.read_csv(csv_file_path, delimiter=';')
        for pic_name in csv_info['Filename']:
            current_image_path = root_test+'/resized/'+pic_name
            np_image = np.array(Image.open(current_image_path))
            all_test_images.append(np_image)
        for label in csv_info['ClassId']:
            all_test_one_hot_labels.append(label)
        all_test_one_hot_labels = one_hot_encode(all_test_one_hot_labels,43)
        self.X_test = all_test_images
        self.y_test = all_test_one_hot_labels
        logger.info('All test images have been processed')
        print('Datasets for test created.')


    def next_batch(self, batch_size, pic_dim):
        batch = self.X_train[self.i: self.i + batch_size]
        if (len(batch) % batch_size == 0):
            x = self.X_train[self.i: self.i + batch_size].reshape(batch_size, pic_dim, pic_dim, 3)
            y = self.y_train[self.i: self.i + batch_size]
            self.i = (self.i + batch_size) % len(self.X_train)
        else:
            x = self.X_train[self.i: self.i + batch_size].reshape(-1, pic_dim, pic_dim, 3)
            y = self.y_train[self.i: self.i + batch_size]
            self.i = (self.i + batch_size) % len(self.X_train)
        return x, y


    def next_batch_test(self, batch_size, pic_dim):
        batch = self.X_test[self.i_test: self.i_test + batch_size]
        if (len(batch) % batch_size == 0):
            x = np.array(self.X_test[self.i_test: self.i_test + batch_size]).reshape(batch_size, pic_dim, pic_dim, 3)
            y = np.array(self.y_test[self.i_test: self.i_test + batch_size])
            self.i_test = (self.i_test + batch_size) % len(self.X_test)
        else:
            x = np.array(self.X_test[self.i_test: self.i_test + batch_size]).reshape(-1, pic_dim, pic_dim, 3)
            y = np.array(self.y_test[self.i_test: self.i_test + batch_size])
            self.i_test = (self.i_test + batch_size) % len(self.X_test)
        return x, y

    # TODO: make the batches method
