import cv2
import numpy as np
from imgaug import augmenters as iaa
from tensorflow.python.keras.preprocessing.image import img_to_array as img_to_array


def createAugmentator():
    seq = iaa.Sequential(
        [iaa.SomeOf((0, 3),
                    [
                        iaa.GaussianBlur((0, 0.95)),  # blur images with a sigma of 0 to 3.0
                        iaa.Sharpen(alpha=(0, 0.10), lightness=(0.85, 1.25)),  # sharpen images
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.025 * 255), per_channel=False),
                        iaa.Multiply((0.8, 1.1), per_channel=False)
                    ]
                    )
         ],
        random_order=True
    )

    return seq

def load_image(image_path):
    seq = createAugmentator()
    img = cv2.imread(image_path)
    img = cv2.resize(img,(64,64))
    do_aug = np.random.randint(0, 2)
    #if do_aug:
    #    img = seq.augment_image(img)

    return img_to_array(img / 255)
