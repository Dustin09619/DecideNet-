import pandas as pd
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Conv2D, Add, Concatenate
from keras.models import Model
from keras.optimizers import Adam
import scipy.io
import os
import cv2
import keras.backend as K
import matplotlib.pyplot as plt

# Paths
dataset_path = '/content/extracted_mall_dataset/mall_dataset/'
images_folder = os.path.join(dataset_path, 'frames')
annotations_path = os.path.join(dataset_path, 'mall_gt.mat')

# Image dimensions
wid = 256
heigh = 192

# Load annotations from .mat file
def load_annotations(annotations_path):
    data = scipy.io.loadmat(annotations_path)
    return data['gt']  # Adjust this based on the actual key in the .mat file

# Normalize image size
def normalize_image_size(img):
    imgwidth = img.shape[1]
    imgheight = img.shape[0]
    assert imgwidth <= 1024
    imgheightleft = 1024 - imgheight
    imgwidthleft = 1024 - imgwidth
    img = np.pad(img, [(imgheightleft // 2, imgheightleft - imgheightleft // 2), 
                       (imgwidthleft // 2, imgwidthleft - imgwidthleft // 2)], 'constant')
    return img

# Data preparation for training
def data_pre_train(images_folder, annotations):
    print('Loading training data...')
    img_names = os.listdir(images_folder)
    img_num = len(img_names)
    
    train_data = []
    for i in range(img_num):
        if i % 10 == 0:
            print(i, '/', img_num)
        name = img_names[i]
        img = cv2.imread(os.path.join(images_folder, name), 0)
        img = np.array(img)
        img = (img - 127.5) / 128
        
        # Get ground truth density from annotations
        img_idx = int(name.split('.')[0])  # Assuming image names are indexed as 1.jpg, 2.jpg, ...
        den = annotations[0][img_idx - 1]  # Adjust based on the structure of the annotations
        
        # Append normalized image and density map
        train_data.append([img, den])
    
    print('Load data finished.')
    return train_data

# Data preparation for testing (if needed)
def data_pre_test(images_folder, annotations):
    print('Loading test data...')
    img_names = os.listdir(images_folder)
    img_num = len(img_names)

    data = []
    for i in range(img_num):
        if i % 50 == 0:
            print(i, '/', img_num)
        name = img_names[i]
        img = cv2.imread(os.path.join(images_folder, name), 0)
        img = np.array(img)
        img = (img - 127.5) / 128
        
        img_idx = int(name.split('.')[0])
        den = annotations[0][img_idx - 1]
        
        data.append([img, den])
    
    print('Load data finished.')
    return data

# Load training annotations
annotations = load_annotations(annotations_path)

# Prepare training data
data = data_pre_train(images_folder, annotations)
np.random.shuffle(data)

# Prepare input and output arrays
x_train = []
y_train = []
for d in data:
    x_train.append(np.reshape(d[0], (d[0].shape[0], d[0].shape[1], 1)))
    y_train.append(np.reshape(d[1], (d[1].shape[0], d[1].shape[1], 1)))

x_train = np.array(x_train)
y_train = np.array(y_train)

# Visualize an example
plt.imshow(x_train[0].reshape(wid, heigh), cmap='gray')
plt.imshow(y_train[0].reshape(wid, heigh), alpha=0.7, cmap='jet')
plt.show()

# Custom metrics
def maaae(y_true, y_pred):
    s = K.sum(K.sum(y_true, axis=1), axis=1)
    s1 = K.sum(K.sum(y_pred, axis=1), axis=1)
    return K.mean(abs(s - s1))

def mssse(y_true, y_pred):
    s = K.sum(K.sum(y_true, axis=1), axis=1)
    s1 = K.sum(K.sum(y_pred, axis=1), axis=1)
    return K.mean((s - s1) * (s - s1))

def customLoss(y_true, y_pred):
    loss1 = mssse(y_true, y_pred)
    loss2 = K.mean((y_true - y_pred)**2)
    return 0.7 * loss2 + 0.3 * loss1

# Model architecture
inputs_1 = Input(shape=(None, None, 1))
conv_s = Conv2D(20, (7, 7), padding='same', activation='relu')(inputs_1)
conv_s = Conv2D(40, (5, 5), padding='same', activation='relu')(conv_s)
conv_s = Conv2D(20, (5, 5), padding='same', activation='relu')(conv_s)
conv_s = Conv2D(10, (5, 5), padding='same', activation='relu')(conv_s)
conv_regnet = Conv2D(1, (1, 1), padding='same')(conv_s)

# Define the model
model = Model(inputs=[inputs_1], outputs=conv_regnet)
model.summary()

# Compile model
adam = Adam(lr=5e-3)
model.compile(loss=customLoss, optimizer=adam, metrics=[maaae, mssse])

# Fit model
model.fit(x_train, y_train, epochs=2000, batch_size=32)
model.save('models/mcnn_mall.h5')
