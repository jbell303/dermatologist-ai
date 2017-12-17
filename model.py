from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    image_files = np.array(data['filenames'])
    image_targets = np_utils.to_categorical(np.array(data['target']), len(data['target']))
    return image_files, image_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('data/train')
valid_files, valid_targets = load_dataset('data/valid')
test_files, test_targets = load_dataset('data/test')

# load list of dog names
category_names = [item[12:-1] for item in sorted(glob("data/train/*/"))]

# print statistics about the dataset
#print('There are %d total categories.' % len(category_names))
#print('There are %s total images.\n' % len(np.hstack([train_files, valid_files, test_files])))
#print('There are %d training images.' % len(train_files))
#print('There are %d validation images.' % len(valid_files))
#print('There are %d test images.'% len(test_files))

print(category_names[0])
train_files_short = train_files[:100]

from keras.applications.resnet50 import ResNet50

model = ResNet50(weights='imagenet')

from keras.preprocessing import image
