### Face recognition using CNN

[//]: # (Image References)

[image1]: http://github.com/yashvardhan90/yashvardhan90.github.io/raw/master/img/PCA_CNN/PCA-cnn_9_1.png
[image2]: http://github.com/yashvardhan90/yashvardhan90.github.io/raw/master/img/PCA_CNN/PCA-cnn_15_0.png
[image3]: http://github.com/yashvardhan90/yashvardhan90.github.io/raw/master/img/PCA_CNN/PCA-cnn_15_1.png

```python
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from scipy import ndimage
from PIL import Image
from sklearn import preprocessing
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dropout, Dense, MaxPooling2D, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
```



```python
data = 'yalefaces/'
faces_data = []
class_label = []
for filename in os.listdir(data):
    if filename.endswith(".txt") or filename.endswith(".DS_Store") :
        continue
    else:
        #extractFrames(filename, 'data_faces')
        img_array = np.array(Image.open(data+filename)).astype(float)
        faces_data.append(img_array)
        label = filename[7:9]
        class_label.append(label)
faces_data = np.array(faces_data)
class_label = np.array(class_label)
```


```python
nb_class = len(np.unique(class_label))

```


```python
lb = preprocessing.LabelBinarizer()
class_label = lb.fit_transform(class_label)
```


```python
print(class_label)
```

    [[0 0 0 ..., 0 0 0]
     [1 0 0 ..., 0 0 0]
     [0 0 0 ..., 0 0 0]
     ..., 
     [0 0 0 ..., 0 0 0]
     [0 0 0 ..., 1 0 0]
     [0 0 0 ..., 0 0 0]]



```python
def normalize(img):
    return 1/255.*(img)

def crop(img):
    return img[:, 50:300]
```


```python
X = []
for img in faces_data:
    #img = crop(img)
    img = normalize(img)
    img = cv2.resize(img, (160,121))
    X.append(img)
X = np.array(X)
print(X.shape)
```

    (166, 121, 160)



```python
from sklearn.utils import shuffle
X, y = shuffle(X, class_label, random_state=0)
```


```python
plt.figure()
plt.imshow(X[10], cmap='gray')
```




    <matplotlib.image.AxesImage at 0x119c7ce80>




![alt text][image1]



```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# For input to Keras model: (batch size, height, width, channels)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
```

    (111, 121, 160, 1) (111, 15)
    (55, 121, 160, 1) (55, 15)



```python
model = Sequential()

model.add(Conv2D(16, (3,3), strides=(1,1), activation='elu', input_shape=(X_train.shape[1],X_train.shape[2],1)))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Conv2D(32, (3,3), activation='elu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Conv2D(32, (3,3), strides=(1,1), activation='elu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='elu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='elu'))
model.add(Dropout(0.3))
model.add(Dense(nb_class, activation='softmax' ))

model.summary()

model.compile(optimizer='Adam',
          loss='categorical_crossentropy',
          metrics=['accuracy'])
#model.fit(X_train, y_train, epochs=8, batch_size=32)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_4 (Conv2D)            (None, 119, 158, 16)      160       
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 59, 79, 16)        0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 57, 77, 32)        4640      
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 28, 38, 32)        0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 26, 36, 32)        9248      
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 13, 18, 32)        0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 7488)              0         
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 7488)              0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 512)               3834368   
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 128)               65664     
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 15)                1935      
    =================================================================
    Total params: 3,916,015
    Trainable params: 3,916,015
    Non-trainable params: 0
    _________________________________________________________________


### Question 2 (f)
### Data Augmentation


```python
# Data Augmentation
from keras.preprocessing.image import ImageDataGenerator
EPOCHS = 30
BATCH_SIZE = 32
train_datagen = ImageDataGenerator(shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

validation_generator = test_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE)

#earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')

history = model.fit_generator(train_generator,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data = validation_generator,
    validation_steps = len(X_test) // BATCH_SIZE,
    verbose=1)
    #callbacks=[earlyStopping])

model.save('model.h5')
print("Model saved!")
```

    Epoch 1/30
    4/3 [========================================] - 3s 861ms/step - loss: 3.4328 - acc: 0.0575 - val_loss: 2.9319 - val_acc: 0.0182
    Epoch 2/30
    4/3 [========================================] - 3s 686ms/step - loss: 2.9228 - acc: 0.1403 - val_loss: 2.5272 - val_acc: 0.1818
    Epoch 3/30
    4/3 [========================================] - 3s 749ms/step - loss: 2.4885 - acc: 0.1838 - val_loss: 2.4730 - val_acc: 0.1636
    Epoch 4/30
    4/3 [========================================] - 2s 611ms/step - loss: 2.2977 - acc: 0.3136 - val_loss: 2.2694 - val_acc: 0.3273
    Epoch 5/30
    4/3 [========================================] - 3s 742ms/step - loss: 1.9934 - acc: 0.4288 - val_loss: 2.0563 - val_acc: 0.3636
    Epoch 6/30
    4/3 [========================================] - 3s 702ms/step - loss: 1.8335 - acc: 0.4747 - val_loss: 1.7684 - val_acc: 0.6000
    Epoch 7/30
    4/3 [========================================] - 2s 607ms/step - loss: 1.6105 - acc: 0.5661 - val_loss: 1.5527 - val_acc: 0.5091
    Epoch 8/30
    4/3 [========================================] - 2s 608ms/step - loss: 1.2896 - acc: 0.5948 - val_loss: 1.3767 - val_acc: 0.5636
    Epoch 9/30
    4/3 [========================================] - 2s 620ms/step - loss: 1.2371 - acc: 0.5931 - val_loss: 1.2162 - val_acc: 0.6364
    Epoch 10/30
    4/3 [========================================] - 2s 594ms/step - loss: 1.2278 - acc: 0.5733 - val_loss: 1.1651 - val_acc: 0.6727
    Epoch 11/30
    4/3 [========================================] - 2s 613ms/step - loss: 1.0793 - acc: 0.5442 - val_loss: 1.0447 - val_acc: 0.6909
    Epoch 12/30
    4/3 [========================================] - 2s 599ms/step - loss: 1.0326 - acc: 0.6948 - val_loss: 1.2848 - val_acc: 0.6182
    Epoch 13/30
    4/3 [========================================] - 2s 600ms/step - loss: 0.8976 - acc: 0.7161 - val_loss: 1.1182 - val_acc: 0.7091
    Epoch 14/30
    4/3 [========================================] - 2s 602ms/step - loss: 0.8456 - acc: 0.7082 - val_loss: 0.9046 - val_acc: 0.7273
    Epoch 15/30
    4/3 [========================================] - 3s 753ms/step - loss: 0.7381 - acc: 0.7822 - val_loss: 0.8139 - val_acc: 0.7636
    Epoch 16/30
    4/3 [========================================] - 3s 775ms/step - loss: 0.5814 - acc: 0.8190 - val_loss: 0.8724 - val_acc: 0.7455
    Epoch 17/30
    4/3 [========================================] - 2s 616ms/step - loss: 0.6171 - acc: 0.7670 - val_loss: 0.7576 - val_acc: 0.7455
    Epoch 18/30
    4/3 [========================================] - 3s 667ms/step - loss: 0.7284 - acc: 0.7728 - val_loss: 0.5410 - val_acc: 0.8909
    Epoch 19/30
    4/3 [========================================] - 3s 630ms/step - loss: 0.5744 - acc: 0.8396 - val_loss: 0.6037 - val_acc: 0.8727
    Epoch 20/30
    4/3 [========================================] - 2s 617ms/step - loss: 0.4618 - acc: 0.8837 - val_loss: 0.6635 - val_acc: 0.8364
    Epoch 21/30
    4/3 [========================================] - 3s 684ms/step - loss: 0.4439 - acc: 0.8225 - val_loss: 0.5555 - val_acc: 0.8545
    Epoch 22/30
    4/3 [========================================] - 2s 616ms/step - loss: 0.3823 - acc: 0.8843 - val_loss: 0.6592 - val_acc: 0.8364
    Epoch 23/30
    4/3 [========================================] - 2s 615ms/step - loss: 0.4111 - acc: 0.8575 - val_loss: 0.7290 - val_acc: 0.8364
    Epoch 24/30
    4/3 [========================================] - 2s 613ms/step - loss: 0.4160 - acc: 0.8541 - val_loss: 0.5520 - val_acc: 0.8545
    Epoch 25/30
    4/3 [========================================] - 2s 619ms/step - loss: 0.3284 - acc: 0.8714 - val_loss: 0.5246 - val_acc: 0.8545
    Epoch 26/30
    4/3 [========================================] - 2s 609ms/step - loss: 0.2572 - acc: 0.8955 - val_loss: 0.6713 - val_acc: 0.8545
    Epoch 27/30
    4/3 [========================================] - 3s 682ms/step - loss: 0.3457 - acc: 0.8719 - val_loss: 0.5676 - val_acc: 0.8727
    Epoch 28/30
    4/3 [========================================] - 3s 646ms/step - loss: 0.3611 - acc: 0.8961 - val_loss: 0.4857 - val_acc: 0.9091
    Epoch 29/30
    4/3 [========================================] - 2s 622ms/step - loss: 0.2753 - acc: 0.9229 - val_loss: 0.5467 - val_acc: 0.8727
    Epoch 30/30
    4/3 [========================================] - 3s 662ms/step - loss: 0.3308 - acc: 0.8648 - val_loss: 0.5625 - val_acc: 0.8727
    Model saved!



```python
print(history.history.keys())
```

    dict_keys(['acc', 'val_acc', 'val_loss', 'loss'])



```python
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```


![alt text][image2]



![alt text][image3]

