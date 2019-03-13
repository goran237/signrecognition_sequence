
import os
import pandas as pd
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('log_file.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

common_dim = 48

def resize_image(path, common_dim):
    image = Image.open(path)
    height = image.height
    width = image.width
    smaller_dim = np.minimum(height, width)
    processed_im = image
    if height != width:
        processed_im = image.crop((0, 0, smaller_dim, smaller_dim))
    if smaller_dim != common_dim:
        processed_im = processed_im.resize((common_dim, common_dim))
    return processed_im


def process_signs_in_folder(folder_name):
    root = './utils/data/GTSRB/Final_Training/Images'
    csv_file_path = root + '/' + folder_name + '/' + 'GT-' + folder_name + '.csv'
    csv_info = pd.read_csv(csv_file_path, delimiter=';')
    folder_resize = root + '/' + folder_name + '_R/'
    if not os.path.exists(folder_resize):
        os.makedirs(folder_resize)
        logger.info('Folder ' + folder_resize + 'created')
        print('Processing training images...')
        for pic_name in csv_info['Filename']:
            current_image_path = root + '/' + folder_name + '/' + pic_name
            current_image = resize_image(current_image_path, common_dim)
            current_image.save(folder_resize + pic_name, 'PPM')

        logger.info('Processed images created in '+folder_resize)
    else:
        logger.info('Folder ' + folder_resize + ' already exists')


def preprocess_training_images():
    for i in range(43):
        folder_name = format(i, '05d')
        process_signs_in_folder(folder_name)

def preprocess_test_images():
    root = './utils/data/GTSRB/Final_Test/Images'
    csv_file_path =root+'/'+'GT-final_test.test.csv'
    csv_info = pd.read_csv(csv_file_path,delimiter=';')
    folder_resize = root+'/resized/'
    if not os.path.exists(folder_resize):
        os.makedirs(folder_resize)
        logger.info('Folder ' + folder_resize + ' created')
        print('Processing all test images...')
        for pic_name in csv_info['Filename']:
            current_image_path = root + '/' + pic_name
            current_image = resize_image(current_image_path,common_dim)
            current_image.save(folder_resize+pic_name, 'PPM')
        logger.info('Processed images created in '+folder_resize)
    else:
        logger.info('Folder ' + folder_resize + ' already exists')

