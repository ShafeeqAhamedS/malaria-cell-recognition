# <p align="center">Deep Neural Network for Malaria Infected Cell Recognition</p>

## AIM:

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset:
Malaria dataset of 27,558 cell images with an equal number of parasitized and uninfected cells. A level-set based algorithm was applied to detect and segment the red blood cells. The images were collected and annotated by medical professionals.Here we build a convolutional neural network model that is able to classify the cells.
<br>
![kv](https://user-images.githubusercontent.com/75235334/193736032-b5847f1f-f002-4edc-912a-eaf48444f1b0.jpg)

## Neural Network Model:
![image](https://github.com/ShafeeqAhamedS/malaria-cell-recognition/assets/93427237/af999033-dd17-401f-94fc-2cd729f1d169)
<br>
## DESIGN STEPS:

### STEP 1:
Import tensorflow and preprocessing libraries.
### STEP 2:
Download and load the dataset folder.
### STEP 3:
Split the  training and testing folders.
### STEP 4:
Perform image data generation methods.
### STEP 6:
Build the convolutional neural network model.
### STEP 7:
Train the model with the training data.
### STEP 8:
Plot the performance plot.
### STEP 9:
Evaluate the model with the testing data using probability prediction(uninfected-> prob>0.5,parasitized-> <=0.5).
### STEP 10:
Fit the model and predict the sample input.

## PROGRAM
### Import Liraries
```py
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1.keras.backend import set_session
```
### Allow GPU Processing
```py
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
```
### Read Images
```py
my_data_dir = "./cell_images"

os.listdir(my_data_dir)

test_path = my_data_dir+'/test/'

train_path = my_data_dir+'/train/'

os.listdir(train_path)

len(os.listdir(train_path+'/uninfected/'))

len(os.listdir(train_path+'/parasitized/'))

os.listdir(train_path+'/parasitized')[0]

para_img= imread(train_path+'/parasitized/'+os.listdir(train_path+'/parasitized')[0])

plt.imshow(para_img)
```
### Checking the image dimensions
```py
dim1 = []
dim2 = []

for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

sns.jointplot(x=dim1,y=dim2)

image_shape = (130,130,3)
```
### Image Generator
```py
help(ImageDataGenerator)

image_gen = ImageDataGenerator(rotation_range=20,         # rotate the image 20 degrees
                               width_shift_range=0.10,    # Shift the pic width by a max of 5%
                               height_shift_range=0.10,   # Shift the pic height by a max of 5%
                               rescale=1/255,             # Rescale the image by normalzing it.
                               shear_range=0.1,           # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1,            # Zoom in by 10% max
                               horizontal_flip=True,      # Allo horizontal flipping
                               fill_mode='nearest'        # Fill in missing pixels with the nearest filled value
                              )

image_gen.flow_from_directory(train_path)

image_gen.flow_from_directory(test_path)
```
### DL Model - Build & Compile
```py
model = models.Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128))
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

batch_size = 16

help(image_gen.flow_from_directory)
```
### Fit the model
```py
train_image_gen = image_gen.flow_from_directory(train_path,
                                                target_size=image_shape[:2],
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='binary')

train_image_gen.batch_size

len(train_image_gen.classes)

train_image_gen.total_batches_seen

test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)

train_image_gen.class_indices

results = model.fit(train_image_gen,epochs=2,validation_data=test_image_gen)

model.save('cell_model.h5')
```
### Plot the graphs
```py
losses = pd.DataFrame(model.history.history)

losses[['loss','val_loss']].plot()

model.metrics_names
```
### Evaluate Metrics
```py
model.evaluate(test_image_gen)

pred_probabilities = model.predict(test_image_gen)

test_image_gen.classes

predictions = pred_probabilities > 0.5

print(classification_report(test_image_gen.classes,predictions))

confusion_matrix(test_image_gen.classes,predictions)
```
### Check for New Image
```py
import random
import tensorflow as tf

list_dir=["Un Infected","parasitized"]
dir_=(random.choice(list_dir))
para_img= imread(train_path+'/'+dir_+'/'+os.listdir(train_path+'/'+dir_)[random.randint(0,100)])

img  = tf.convert_to_tensor(np.asarray(para_img))
img = tf.image.resize(img,(130,130))

img=img.numpy()
pred=bool(model.predict(img.reshape(1,130,130,3))<0.5 )

plt.title("Model prediction: "+("Parasitized" if pred  else "Un Infected")+"\nActual Value: "+str(dir_))
plt.axis("off")
plt.imshow(img)
plt.show()
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/ShafeeqAhamedS/malaria-cell-recognition/assets/93427237/a225ad9d-ab7e-49b0-9111-60f004c904b5)
### Classification Report
![image](https://github.com/ShafeeqAhamedS/malaria-cell-recognition/assets/93427237/d8a44eed-9876-4c8d-8abf-2072680519be)
### Confusion Matrix
![image](https://github.com/ShafeeqAhamedS/malaria-cell-recognition/assets/93427237/b849bc4c-24bb-468e-8106-20e4439e45fc)
### New Sample Data Prediction
![image](https://github.com/ShafeeqAhamedS/malaria-cell-recognition/assets/93427237/f1db77e8-e904-47bd-bd1c-2dc0e7e49a38)
## RESULT:
Thus, a deep neural network for Malaria infected cell recognition is developed and the performance is analyzed.
