import os
import pickle

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import requests
from zipfile import ZipFile
import sys

if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # if not os.path.exists('../ImageData'):
    #     os.mkdir('../ImageData')

    if not os.path.exists('../SavedModels'):
        os.mkdir('../SavedModels')

    if not os.path.exists('../SavedHistory'):
        os.mkdir('../SavedHistory')

    # Download data if it is unavailable.
    if 'cats-and-dogs-images.zip' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Image dataset is loading.\n")
        url = "https://www.dropbox.com/s/jgv5zpw41ydtfww/cats-and-dogs-images.zip?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/cats-and-dogs-images.zip', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")

        sys.stderr.write("\n[INFO] Extracting files.\n")
        with ZipFile('../Data/cats-and-dogs-images.zip', 'r') as zip:
            zip.extractall(path="../Data")
            sys.stderr.write("[INFO] Completed.\n")

    # write your code here

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_set = datagen.flow_from_directory(target_size=(150, 150), directory="../Data/", classes=["test"], shuffle=False)


model = load_model("../SavedModels/stage_two_model.h5")
prediction = model.predict(test_set)
with open('../SavedHistory/stage_three_history', 'wb') as file_pi:
    pickle.dump(prediction, file_pi)
#np.argmax(prediction, axis=1),


#print(f"{test_set.target_size[0]} {test_set.target_size[1]} {training_set.batch_size} {test_set.shuffle}")
