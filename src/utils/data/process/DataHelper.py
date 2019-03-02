import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt

class DataHelper:

    def __init__(self):
        self.i = 0
        self.training_images = []
        self.training_labels = []

    def set_up_images(self):
        root = './utils/data/GTSRB/Final_Training/Images'
        for i in range (43):
            folder_name = format(i, '05d')
            folder_name_resized = folder_name + '_R'
            csv_file_path = root + '/' + folder_name + '/' + 'GT-' + folder_name + '.csv'
            csv_info = pd.read_csv(csv_file_path, delimiter=';')
            for pic_name in csv_info['Filename']:
                current_image_path = root + '/' + folder_name_resized + '/' + pic_name
                np_image = np.array(Image.open(current_image_path))
                norm_image = np.array((np_image-np.min(np_image))/(np.max(np_image)-np.min(np_image)))
                self.training_images.append(norm_image)

    #TODO: randomize images
    #TODO: make the batches method

        print('All images collected')





