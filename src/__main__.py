from src.utils.data.importer.DataImporter import import_training_data,import_test_data
from src.utils.data.process.DataProcessor import preprocess_training_images,preprocess_test_images
from src.train.Trainer import perform_train
import sys

from test.Tester import perform_test


def main():
    import_training_data()
    import_test_data()
    preprocess_training_images()
    preprocess_test_images()
    perform_train()
    perform_test()


if __name__ == '__main__':
    main()
