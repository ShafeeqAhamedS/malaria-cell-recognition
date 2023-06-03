# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

### STEP 2:

### STEP 3:

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
### Check for custom image
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

### Classification Report

### Confusion Matrix

### New Sample Data Prediction

## RESULT
