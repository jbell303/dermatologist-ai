# USAGE
# python train_network.py --dataset images --model santa_not_santa.model

# import python modules for use with pythonw
import sys
sys.path.append('/Users/jbell303/anaconda3/envs/derm-ai/lib/python3.5/site-packages')

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.datasets import load_files       
from keras.utils import np_utils
from keras.preprocessing import image   
from pyimagesearch.lenet import LeNet
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True    

import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 25
INIT_LR = 1e-3
BS = 32

# initialize the data and labels
print("[INFO] loading images...")

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    image_files = np.array(data['filenames'])
    image_targets = np_utils.to_categorical(np.array(data['target']), num_classes=3)
    return image_files, image_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('data/train')
valid_files, valid_targets = load_dataset('data/valid')

category_names = [item[11:-1] for item in sorted(glob("data/train/*/"))]

print('[DEBUG] train target shape: {}'.format(train_targets.shape))
print('[DEBUG] valid target shape: {}'.format(valid_targets.shape))

# convert image paths to tensors
def path_to_image(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(28, 28))
    # convert PIL.Image.Image type to 3D tensor with shape (28, 28, 3)
    return image.img_to_array(img)

def paths_to_image(img_paths):
    return [path_to_image(img_path) for img_path in tqdm(img_paths)]
   

train_images = np.array(paths_to_image(train_files), dtype='float') / 255
valid_images = np.array(paths_to_image(valid_files), dtype='float') / 255

# print statistics about the dataset
print('[INFO] There are %d total categories.' % len(category_names))
print('[INFO] There are %s total images.\n' % len(np.hstack([train_files, valid_files])))
print('[INFO] There are %d training images.' % len(train_files))
print('[INFO] There are %d validation images.' % len(valid_files))

print('[DEBUG] TrainX shape: {}, TrainY shape: {}, TestX shape: {}, TestY shape: {}'
	.format(train_images.shape, train_targets.shape, valid_images.shape, valid_targets.shape))

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=3, classes=3)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
model.summary()

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(train_images, train_targets, batch_size=BS),
	validation_data=(valid_images, valid_targets), steps_per_epoch=len(train_images) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dematologist-AI")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])