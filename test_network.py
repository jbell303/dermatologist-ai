# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

import sys
sys.path.append('/Users/jbell303/anaconda3/envs/derm-ai/lib/python3.5/site-packages')
sys.path.append('/Users/jbell303/anaconda3/envs/aind-dog/lib/python3.5/site-packages/cv2')

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
from imutils import paths
import cv2
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-d", "--dataset", required=True,
	help="path to test dataset")
ap.add_argument("-o", "--output", default='output.csv',
	help="path to output csv file")
args = vars(ap.parse_args())

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# open the csv file
with open(args["output"], 'w') as csvfile:
	filewriter = csv.writer(csvfile, delimiter=',',
		quotechar='|', quoting=csv.QUOTE_MINIMAL)
	filewriter.writerow(['Id', 'task_1', 'task_2'])

	# grab the test images
	imagePaths = sorted(list(paths.list_images(args["dataset"])))
	num_rows = 0
	for path in imagePaths:
		# load the image
		image = cv2.imread(path)

		# pre-process the image for classification
		image = cv2.resize(image, (28, 28))
		image = image.astype("float") / 255.0
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)

		# classify the input image
		pred = model.predict(image)[0]
		print("[DEBUG] file: {}, prediction: {}".format(path, pred))
		filewriter.writerow([path, pred[0], pred[2]])
		num_rows += 1

print("[DEBUG] wrote {} rows to file: {}".format(num_rows, args["output"]))
						