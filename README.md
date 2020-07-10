# Image-Super-Resolution-using-Convolution-Neural-Networks-and-Auto-encoders
Enhancing image quality with Deep Learning and CNN 

# Implementation:

Library Imports

```python
import numpy as np
import cv2
import glob
import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import pickle
```
# Dataset
We will be working on ‘Labeled Faces in the Wild Home’ dataset. This dataset contains a database of labelled faces, generally used for face recognition and detection. However, our aim is not to detect faces but to make a model to enhance image resolution.

```python
face_images = glob.glob('lfw/lfw/**/*.jpg') #returns path of images
print(len(face_images)) #contains 13243 images
``` 
# Load and Preprocess Images

The size of original images are of 250 x 250 pixels. However, it would take a lot computation power to process these images on normal computer. Therefore, we will reduce the size of all images to 80 x 80 pixels.

    As there are around 13,000 images, it would take lot of time if we process it individually. Hence, we take advantage of multiprocessing library provided in python for ease of execution.

    tqdm is a progress library that we use to get a progress bar of the work done.
    

```python
from tqdm import tqdm
from multiprocessing import Poolprogress = tqdm(total= len(face_images), position=0)
def read(path):
  img = image.load_img(path, target_size=(80,80,3))
  img = image.img_to_array(img)
  img = img/255.
  progress.update(1)
  return imgp = Pool(10)
img_array = p.map(read, face_images)
```

# Data preparation

 we will split our dataset to train and validation set. We will use train data to train our model and validation data will be used to evaluate the model.
 
```python
all_images = np.array(img_array)#Split test and train data. all_images will be our output images
train_x, val_x = train_test_split(all_images, random_state = 32, test_size=0.2)
```

As this is an image resolution enhancement task we will distort our images and take it as an input images. The original images will be added as our output images.
```python
#now we will make input images by lowering resolution without changing the size
def pixalate_image(image, scale_percent = 40):
  width = int(image.shape[1] * scale_percent / 100)
  height = int(image.shape[0] * scale_percent / 100)
  dim = (width, height)  small_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  
  # scale back to original size
  width = int(small_image.shape[1] * 100 / scale_percent)
  height = int(small_image.shape[0] * 100 / scale_percent)
  dim = (width, height)  low_res_image = cv2.resize(small_image, dim, interpolation =  cv2.INTER_AREA) return low_res_image
```

The idea is to take these distorted images and feed it to our model and make model learn to get the original image back.
```python
train_x_px = []for i in range(train_x.shape[0]):
  temp = pixalate_image(train_x[i,:,:,:])
  train_x_px.append(temp)train_x_px = np.array(train_x_px)   #Distorted images# get low resolution images for the validation set
val_x_px = []for i in range(val_x.shape[0]):
  temp = pixalate_image(val_x[i,:,:,:])
  val_x_px.append(temp)val_x_px = np.array(val_x_px)     #Distorted images
```  
 
<p align="center">
  <img width="100" height="100" src= https://miro.medium.com/max/251/1*6vollQxPsSu07FZg1X9_2g.png>
</p>
<p align="center">
  Input Image 
</p>

<p align="center">
  <img width="100" height="100" src= https://miro.medium.com/max/251/1*R-thvW_0gVh9Y79lBlYEZw.png>
  
  
</p>

<p align="center">
  Originale Image
</p>




# Model building

Let's define the structure of model. Moreover, to overcome the possibility of over-fitting, we are using l1 regularization technique in our convolution layer.
```python
Input_img = Input(shape=(80, 80, 3))  
    
#encoding architecture
x1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(Input_img)
x2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x1)
x3 = MaxPool2D(padding='same')(x2)x4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x3)
x5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x4)
x6 = MaxPool2D(padding='same')(x5)encoded = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x6)
#encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)# decoding architecture
x7 = UpSampling2D()(encoded)
x8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x7)
x9 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x8)
x10 = Add()([x5, x9])x11 = UpSampling2D()(x10)
x12 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x11)
x13 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x12)
x14 = Add()([x2, x13])# x3 = UpSampling2D((2, 2))(x3)
# x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
# x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
decoded = Conv2D(3, (3, 3), padding='same',activation='relu', kernel_regularizer=regularizers.l1(10e-10))(x14)autoencoder = Model(Input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

```

# Model Training

We will first define some callbacks so that it would be easy for model visualization and evaluation in future.
```python
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1, mode='min')model_checkpoint =  ModelCheckpoint('superResolution_checkpoint3.h5', save_best_only = True)
```
Let's train our model:
```python
history = autoencoder.fit(train_x_px,train_x,
            epochs=500,
            validation_data=(val_x_px, val_x),
            callbacks=[early_stopper, model_checkpoint])
 
```

    The execution time was around 21 seconds per epoch on 12GB NVIDIA Tesla K80 GPU. EarlyStopping was achieved at 65th epoch.

Now, let's evaluate our model on our test dataset:
```python
results = autoencoder.evaluate(val_x_px, val_x)
print('val_loss, val_accuracy', results)
```
    val_loss, val_accuracy [0.002111854264512658, 0.9279356002807617]

We are getting some pretty good results from our model with around 93% validation accuracy and a validation loss of 0.0021.
# Make Predictions
```python
predictions = autoencoder.predict(val_x_px)n = 4
plt.figure(figsize= (20,10))for i in range(n):
  ax = plt.subplot(3, n, i+1)
  plt.imshow(val_x_px[i+20])
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)  ax = plt.subplot(3, n, i+1+n)
  plt.imshow(predictions[i+20])
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)plt.show()
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
