import matplotlib.pyplot as plt
import numpy as np
import os
#import tqdm
from keras.models import load_model

import itertools
import cv2


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:
            val = cm[i, j]
        else:
            val = int(cm[i, j])

        plt.text(j, i, val,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    dir_dst = './confusion_matrix'
    try:
        os.makedirs(dir_dst)
    except:
        pass
    conf_matrix_name = '{}/{}'.format(dir_dst, '/confusion_matrix.png')
    plt.savefig(conf_matrix_name)

def get_wrongly_classified(images, labels, model):
    dst = './model_evaluation'
    count = 0
    idx = 0
    for img in images:
        p_char = model.predict(img[None, ...])[0]
        pred_char = p_char.argmax()

        gt_char = labels[idx].argmax()
        idx += 1
        if pred_char != gt_char:
            dir_dst = '{}/{}'.format(dst, pred_char)

            try:
                os.makedirs(dir_dst)
            except:
                pass

            new_fname = '{}-{:0>5d}.png'.format(gt_char, count)
            path_dst = '{}/{}'.format(dir_dst, new_fname)
            img_original = img * 255
            img_original = np.uint8(img_original)
            cv2.imwrite(path_dst, img_original)

            count += 1

def evaluate_model(test_images, labels, label_names, model, batch_size=32):
    nb_classes = 43  # number of classes
    steps_test = int(len(labels) / batch_size)  # num test images / batch size
    confusion_matrix = np.zeros((nb_classes, nb_classes))
    class_names_list = os.listdir('./data/train')
    class_names_dict = {}
    idx = 0
    for class_name in class_names_list:
        class_names_dict[class_name] = idx
        idx += 1

    print('labels are: ', label_names)
    idx = 0
    for step in range(steps_test):
        X_batch_img = test_images[idx:idx + batch_size]
        y_batch_labels = labels[idx:idx + batch_size]

        y_batch_pred = model.predict(X_batch_img)
        idx += batch_size

        for gt, pr in zip(y_batch_labels, y_batch_pred):
            confusion_matrix[gt.argmax(), pr.argmax()] += 1

    plt.rcParams['figure.figsize'] = (15, 15)
    plot_confusion_matrix(confusion_matrix, label_names)

    get_wrongly_classified(test_images, labels, model)