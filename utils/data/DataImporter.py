import os
import requests
import zipfile
import io


def import_data():
    url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
    if not os.path.exists('GTSRB/Final_Training/Images'):
        r = requests.get(url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall('.')


def main():
    print("A")


if __name__ == '__main__':
    main()
