import pandas as pd
import math
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, MaxPooling2D, Reshape, Concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import sys
import os
import cv2
import keras.backend as K
import matplotlib.pyplot as plt

dataset = 'Mall'  # Use 'Mall' to denote the dataset
print('dataset:', dataset)

# Update the dataset paths for the Mall dataset
train_path = '/content/extracted_mall_dataset/train/'
train_den_path = '/content/extracted_mall_dataset/train_den/'
val_path = '/content/extracted_mall_dataset/val/'
val_den_path = '/content/extracted_mall_dataset/val_den/'

wid = 256
heigh = 192

def normalize_image_size(img):
    imgwidth = img.shape[1]
    imgheight = img.shape[0]
    assert imgwidth <= 1024
    imgheightleft = 1024 - imgheight
    imgwidthleft = 1024 - imgwidth
    img = np.pad(img, [(imgheightleft//2,imgheightleft-imgheightleft//2), (imgwidthleft//2, imgwidthleft - imgwidthleft//2)], 'constant')
    return img

def data_pre_train():
    print('loading data from dataset', dataset, '...')
    train_img_names = os.listdir(train_path)
    img_num = len(train_img_names)

    train_data = []
    for i in range(img_num):
        if i % 100 == 0:
            print(i, '/', img_num)
        name = train_img_names[i]
        img = cv2.imread(train_path + name, 0)
        img = np.array(img)
        img = (img - 127.5) / 128
        den = np.loadtxt(open(train_den_path + name[:-4] + '.csv'), delimiter=",")
        den_quarter = np.zeros((int(den.shape[0] / 4), int(den.shape[1] / 4)))
        
        for i in range(len(den_quarter)):
            for j in range(len(den_quarter[0])):
                for p in range(4):
                    for q in range(4):
                        den_quarter[i][j] += den[i * 4 + p][j * 4 + q]
        
        train_data.append([img, den_quarter])

    print('load data finished.')
    return train_data

def data_pre_test():
    print('loading test data from dataset', dataset, '...')
    img_names = os.listdir(val_path)
    img_num = len(img_names)

    data = []
    for i in range(img_num):
        if i % 50 == 0:
            print(i, '/', img_num)
        name = 'IMG_' + str(i + 1) + '.jpg'  # Assuming test images are named as 'IMG_1.jpg', 'IMG_2.jpg', etc.
        img = cv2.imread(val_path + name, 0)
        img = np.array(img)
        img = (img - 127.5) / 128
        den = np.loadtxt(open(val_den_path + name[:-4] + '.csv'), delimiter=",")
        den_quarter = np.zeros((int(den.shape[0] / 4), int(den.shape[1] / 4)))
        
        for i in range(len(den_quarter)):
            for j in range(len(den_quarter[0])):
                for p in range(4):
                    for q in range(4):
                        den_quarter[i][j] += den[i * 4 + p][j * 4 + q]

        data.append([img, den_quarter])

    print('load data finished.')
    return data

data = data_pre_train()
np.random.shuffle(data)

x_train = []
y_train = []
for d in data:
    x_train.append(np.reshape(d[0], (d[0].shape[0], d[0].shape[1], 1)))
    y_train.append(np.reshape(d[1], (d[1].shape[0], d[1].shape[1], 1)))

x_train = np.array(x_train)
y_train = np.array(y_train)

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
    loss2 = K.mean((y_true - y_pred) ** 2)
    return 0.7 * loss1 + 0.3 * loss2

inputs = Input(shape=(None, None, 1))

conv_s = Conv2D(24, (5, 5), padding='same', activation='relu')(inputs)
conv_s = MaxPooling2D(pool_size=(2, 2))(conv_s)
conv_s = Conv2D(48, (3, 3), padding='same', activation='relu')(conv_s)
conv_s = MaxPooling2D(pool_size=(2, 2))(conv_s)
conv_s = Conv2D(24, (3, 3), padding='same', activation='relu')(conv_s)
conv_s = Conv2D(12, (3, 3), padding='same', activation='relu')(conv_s)

conv_m = Conv2D(20, (7, 7), padding='same', activation='relu')(inputs)
conv_m = MaxPooling2D(pool_size=(2, 2))(conv_m)
conv_m = Conv2D(40, (5, 5), padding='same', activation='relu')(conv_m)
conv_m = MaxPooling2D(pool_size=(2, 2))(conv_m)
conv_m = Conv2D(20, (5, 5), padding='same', activation='relu')(conv_m)
conv_m = Conv2D(10, (5, 5), padding='same', activation='relu')(conv_m)

conv_l = Conv2D(16, (9, 9), padding='same', activation='relu')(inputs)
conv_l = MaxPooling2D(pool_size=(2, 2))(conv_l)
conv_l = Conv2D(32, (7, 7), padding='same', activation='relu')(conv_l)
conv_l = MaxPooling2D(pool_size=(2, 2))(conv_l)
conv_l = Conv2D(16, (7, 7), padding='same', activation='relu')(conv_l)
conv_l = Conv2D(8, (7, 7), padding='same', activation='relu')(conv_l)

conv_concat3 = Concatenate(axis=3)([conv_m, conv_s, conv_l])
result = Conv2D(1, (1, 1), padding='same')(conv_concat3)

model = Model(inputs=inputs, outputs=result)
model.summary()
reduce_lr = ReduceLROnPlateau(monitor='val_maaae', factor=0.90, cooldown=10, patience=10, min_lr=1e-5)
callbacks_list = [reduce_lr]
adam = Adam(lr=5e-3)
model.compile(loss=customLoss, optimizer=adam, metrics=[maaae, mssse])

list_ = np.arange(len(x_train))
np.random.shuffle(list_)

for i in range(1):
    model.fit(x_train[list_[0:int(0.8*len(list_))]], y_train[list_[0:int(0.8*len(list_))]], epochs=2000, batch_size=32, callbacks=callbacks_list, validation_data=(x_train[list_[int(0.8*len(list_)): ]], y_train[list_[int(0.8*len(list_)):]]))
    model.save('/content/models/mcnn_mall.h5')  # Save the model to the specified path
