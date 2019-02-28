import os
import requests
import zipfile
import io
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt


def import_data():
    url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
    if not os.path.exists('GTSRB/Final_Training/Images'):
        r = requests.get(url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall('.')
        print('Data extracted')


def view_labels_and_images():
    details = pd.read_csv('./GTSRB/Final_Training/Images/00000/GT-00000.csv', delimiter=';')
    currentImage = Image.open('./GTSRB/Final_Training/Images/00000/00000_00000.ppm')
    plt.imshow(currentImage)
    plt.show()


def main():
    view_labels_and_images()


if __name__ == '__main__':
    main()
