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
                #TODO: normalize the images
                self.training_images.append(np_image)
                #TODO: randomize images

        print('All images collected')





