import os
import zipfile


def unzip_data():
    if not os.path.exists('data/test/jpg'):
        print('Extracting test images...')
        z = zipfile.ZipFile('./data/test/test_data.zip')
        z.extractall('./data/test')
        print('Extracted test data.')
        print('--------------------------')
        print('Extracting train images...')
        z = zipfile.ZipFile('./data/train/train_data.zip')
        z.extractall('./data/train')
        print('Extracted train data.')
        print('--------------------------')

def unzip_data_ppm():
    if not os.path.exists('data/test/ppm'):
        print('Extracting test ppm images...')
        z = zipfile.ZipFile('./data/test/test_data_ppm.zip')
        z.extractall('./data/test/ppm')
        print('Extracted test ppm data.')
        print('--------------------------')
    if not os.path.exists('data/train/ppm'):
        print('Extracting train ppm images...')
        z = zipfile.ZipFile('./data/train/train_data_ppm.zip')
        z.extractall('./data/train/ppm')
        print('Extracted train ppm data.')
        print('--------------------------')

def extractData():
    unzip_data()
    unzip_data_ppm()
