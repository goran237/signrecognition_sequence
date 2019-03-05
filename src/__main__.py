from src.utils.data.importer.DataImporter import import_training_data,import_test_data
from src.utils.data.process.DataHelper import DataHelper
from src.utils.data.process.DataProcessor import preprocess_training_images,preprocess_test_images
from src.model.PlaceholderGenerator import generate_placeholders
from src.train.Trainer import perform_train
import sys


def main():
    import_training_data()
    import_test_data()
    preprocess_training_images()
    preprocess_test_images()
    perform_train()


if __name__ == '__main__':
    main()
    sys.exit(main() or 0)
