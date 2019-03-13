import os
import requests
import zipfile
import io
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('log_file.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def import_training_data():
    url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
    if not os.path.exists('utils/data/GTSRB/Final_Training/Images'):
        logger.info('Downloading training images...')
        r = requests.get(url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        logger.info('Extracting images...')
        z.extractall('./utils/data/')
        logger.info('Training data extracted successfully')
    else:
        logger.info('Training data already imported')


def import_test_data():
    url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip'
    url_gt_csv = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip'
    if not os.path.exists('utils/data/GTSRB/Final_Test/Images'):
        logger.info('Downloading test images...')
        r = requests.get(url, stream = True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        logger.info('Extracting images...')
        z.extractall('./utils/data/')
        logger.info('Test data extracted successfully')
    else:
        logger.info('Test data already imported')
    if not os.path.exists('utils/data/GTSRB/Final_Test/Images/GT-final_test.csv'):
        r = requests.get(url_gt_csv, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        logger.info('Extracting ground truth csv...')
        z.extractall('./utils/data/GTSRB/Final_Test/Images/')
        logger.info('CSV extracted successfully')
