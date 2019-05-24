import os

def clean_logs():
    logs_train = './logs/train'
    logs_validation = './logs/validation'
    print('--------------------------')
    print("Deleting previous logs...")
    for the_file in os.listdir(logs_train):
        file_path = os.path.join(logs_train,the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    for the_file in os.listdir(logs_validation):
        file_path = os.path.join(logs_validation,the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    print('--------------------------')