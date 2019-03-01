from src.utils.data.importer.DataImporter import import_data
from src.utils.data.process.DataProcessor import preprocess_images
from utils.data.process.DataHelper import DataHelper
import sys


def main():
    import_data()
    preprocess_images()
    dh = DataHelper()
    dh.set_up_images()


if __name__ == '__main__':
    main()
    sys.exit(main() or 0)
