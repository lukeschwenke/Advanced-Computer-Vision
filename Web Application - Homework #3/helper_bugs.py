import os
import numpy as np
import pickle
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2 , preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input as densenet_preprocess

# Set path to the bug class labels
# Load Pre-Trained Custom CNN Model (jupyter notebook)
current_path = os.getcwd()
bugs_category_path = '/Users/lmschwenke/Downloads/bugs/classes.txt'
predictor_model_cnn = load_model(r'/Users/lmschwenke/Downloads/bugs/cnn_model.h5')
predictor_model_resnet = load_model(r'/Users/lmschwenke/Downloads/bugs/resnet_model.h5')
print("Weights loaded")

# Open the class labels, read each line and assign an integer label
f = open('/Users/lmschwenke/Downloads/bugs/classes.txt')
label = []
name = []
for line in f.readlines():
    label.append(int(line.split()[0]))
    name.append(' '.join(line.split()[1:]))
bug_classes = pd.DataFrame([label, name]).T
bug_classes.columns = ['label','name']
bug_classes.head(3)

input_shape = (64,64,3)
input_layer = Input(shape=input_shape)

print("Models loaded")

# function takes an image [path] provided by the user, converts it to vector, makes the prediction with the 
# already trained model, and stores the predicted value
def predictor(img_path, model_chosen):
    img = load_img(img_path, target_size=(64,64))
    print(img_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    prediction = model_chosen.predict(img)*100 #predictor_model.predict(img)*100
    prediction = pd.DataFrame(np.round(prediction,1), columns = bug_classes.name).transpose()
    prediction.columns = ['values']
    prediction  = prediction.nlargest(5, 'values')
    prediction = prediction.reset_index()
    prediction.columns = ['name', 'values']
    return(prediction)
