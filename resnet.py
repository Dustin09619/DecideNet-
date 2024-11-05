
Open In Colab

%tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
     
TensorFlow 2.x selected.
Found GPU at: /device:GPU:0

import numpy as np
import tensorflow as tf
import h5py
import math

def load_dataset():
    train_dataset = h5py.File('/content/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('/content/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

    classes = np.array(test_dataset["list_classes"][:]) 
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    """
    
    m = X.shape[0]                  
    mini_batches = []
    np.random.seed(seed)
    
    # Shuffle 
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Partition
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    """
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
    Z1 = tf.add(tf.matmul(W1, X), b1)                     
    A1 = tf.nn.relu(Z1)                                   
    Z2 = tf.add(tf.matmul(W2, A1), b2)                   
    A2 = tf.nn.relu(Z2)                                    
    Z3 = tf.add(tf.matmul(W3, A2), b3)                   
    
    return Z3

def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction
     

!pip install graphviz
     
Requirement already satisfied: graphviz in /usr/local/lib/python3.6/dist-packages (0.10.1)

import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.models import Model, load_model

#from tensorflow.python.keras.utils import plot_model
#from utility_functions import *
from tensorflow.python.keras.initializers import glorot_uniform
#import tensorflow.python.keras.backend as K

#K.set_image_data_format('channels_last')
#K.set_learning_phase(1)

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block 
    """

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    
    # Second component of main path
    X = Conv2D(filters= F2,kernel_size= (f,f), strides=(1,1),padding='same',name =conv_name_base+'2b',kernel_initializer= glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name= bn_name_base +'2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3,kernel_size= (1,1),strides=(1,1),padding='valid',name =conv_name_base+ '2c',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3 ,name=bn_name_base +'2c')(X)

    # Adding shortcut value to main path and passing it through a RELU activation
    X = Add()([X_shortcut,X])
    X = Activation('relu')(X)
       
    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block
    """
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a',padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(F2, (f, f), strides = (1,1), name = conv_name_base + '2b',padding='same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c',padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Shortcut path
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1',padding='valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Adding shortcut value to main path and passing it through a RELU activation
    X = Add()([X_shortcut,X])
    X = Activation('relu')(X)
    
    return X

def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Implementation of the  ResNet50 architecture.
    
    """

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f = 3, filters = [128,128,512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128,128,512], stage=3, block='b')
    X = identity_block(X, 3, [128,128,512], stage=3, block='c')
    X = identity_block(X, 3, [128,128,512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f = 3, filters =  [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3,  [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3,  [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2,2),name='avg_pool')(X)
    
    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    # model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

#tf.reset_default_graph()

# model compiling
model = ResNet50(input_shape = (64, 64, 3), classes = 6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
     

train_dataset = h5py.File('/content/train_signs.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:])
train_set_y_orig = np.array(train_dataset["train_set_y"][:])

test_dataset = h5py.File('/content/test_signs.h5', "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

classes = np.array(test_dataset["list_classes"][:]) 

train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

     

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T


     

model.fit(X_train, Y_train, epochs = 20, batch_size = 32)
     
Train on 1080 samples
Epoch 1/20
1080/1080 [==============================] - 6s 6ms/sample - loss: 1.7376 - accuracy: 0.4870
Epoch 2/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.4892 - accuracy: 0.8167
Epoch 3/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.4302 - accuracy: 0.8500
Epoch 4/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.3132 - accuracy: 0.9130
Epoch 5/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.2924 - accuracy: 0.9139
Epoch 6/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.1629 - accuracy: 0.9481
Epoch 7/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.2199 - accuracy: 0.9324
Epoch 8/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.3302 - accuracy: 0.9111
Epoch 9/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.1572 - accuracy: 0.9583
Epoch 10/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.0866 - accuracy: 0.9713
Epoch 11/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.0872 - accuracy: 0.9704
Epoch 12/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.1605 - accuracy: 0.9537
Epoch 13/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.1238 - accuracy: 0.9676
Epoch 14/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.1657 - accuracy: 0.9685
Epoch 15/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.1288 - accuracy: 0.9620
Epoch 16/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.0697 - accuracy: 0.9731
Epoch 17/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.1191 - accuracy: 0.9722
Epoch 18/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.0801 - accuracy: 0.9815
Epoch 19/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.0250 - accuracy: 0.9935
Epoch 20/20
1080/1080 [==============================] - 1s 1ms/sample - loss: 0.0111 - accuracy: 0.9963
<tensorflow.python.keras.callbacks.History at 0x7f040e20a9e8>

preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
     
120/120 [==============================] - 1s 6ms/sample - loss: 0.2791 - accuracy: 0.9250
Loss = 0.2790559738874435
Test Accuracy = 0.925

from tensorflow.keras.utils import plot_model
plot_model(model,to_file='model_architecture.png')

     


model.summary()
     
Model: "ResNet50"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            [(None, 64, 64, 3)]  0                                            
__________________________________________________________________________________________________
zero_padding2d_1 (ZeroPadding2D (None, 70, 70, 3)    0           input_2[0][0]                    
__________________________________________________________________________________________________
conv1 (Conv2D)                  (None, 32, 32, 64)   9472        zero_padding2d_1[0][0]           
__________________________________________________________________________________________________
bn_conv1 (BatchNormalization)   (None, 32, 32, 64)   256         conv1[0][0]                      
__________________________________________________________________________________________________
activation_49 (Activation)      (None, 32, 32, 64)   0           bn_conv1[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 15, 15, 64)   0           activation_49[0][0]              
__________________________________________________________________________________________________
res2a_branch2a (Conv2D)         (None, 15, 15, 64)   4160        max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
bn2a_branch2a (BatchNormalizati (None, 15, 15, 64)   256         res2a_branch2a[0][0]             
__________________________________________________________________________________________________
activation_50 (Activation)      (None, 15, 15, 64)   0           bn2a_branch2a[0][0]              
__________________________________________________________________________________________________
res2a_branch2b (Conv2D)         (None, 15, 15, 64)   36928       activation_50[0][0]              
__________________________________________________________________________________________________
bn2a_branch2b (BatchNormalizati (None, 15, 15, 64)   256         res2a_branch2b[0][0]             
__________________________________________________________________________________________________
activation_51 (Activation)      (None, 15, 15, 64)   0           bn2a_branch2b[0][0]              
__________________________________________________________________________________________________
res2a_branch1 (Conv2D)          (None, 15, 15, 256)  16640       max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
res2a_branch2c (Conv2D)         (None, 15, 15, 256)  16640       activation_51[0][0]              
__________________________________________________________________________________________________
bn2a_branch1 (BatchNormalizatio (None, 15, 15, 256)  1024        res2a_branch1[0][0]              
__________________________________________________________________________________________________
bn2a_branch2c (BatchNormalizati (None, 15, 15, 256)  1024        res2a_branch2c[0][0]             
__________________________________________________________________________________________________
add_16 (Add)                    (None, 15, 15, 256)  0           bn2a_branch1[0][0]               
                                                                 bn2a_branch2c[0][0]              
__________________________________________________________________________________________________
activation_52 (Activation)      (None, 15, 15, 256)  0           add_16[0][0]                     
__________________________________________________________________________________________________
res2b_branch2a (Conv2D)         (None, 15, 15, 64)   16448       activation_52[0][0]              
__________________________________________________________________________________________________
bn2b_branch2a (BatchNormalizati (None, 15, 15, 64)   256         res2b_branch2a[0][0]             
__________________________________________________________________________________________________
activation_53 (Activation)      (None, 15, 15, 64)   0           bn2b_branch2a[0][0]              
__________________________________________________________________________________________________
res2b_branch2b (Conv2D)         (None, 15, 15, 64)   36928       activation_53[0][0]              
__________________________________________________________________________________________________
bn2b_branch2b (BatchNormalizati (None, 15, 15, 64)   256         res2b_branch2b[0][0]             
__________________________________________________________________________________________________
activation_54 (Activation)      (None, 15, 15, 64)   0           bn2b_branch2b[0][0]              
__________________________________________________________________________________________________
res2b_branch2c (Conv2D)         (None, 15, 15, 256)  16640       activation_54[0][0]              
__________________________________________________________________________________________________
bn2b_branch2c (BatchNormalizati (None, 15, 15, 256)  1024        res2b_branch2c[0][0]             
__________________________________________________________________________________________________
add_17 (Add)                    (None, 15, 15, 256)  0           activation_52[0][0]              
                                                                 bn2b_branch2c[0][0]              
__________________________________________________________________________________________________
activation_55 (Activation)      (None, 15, 15, 256)  0           add_17[0][0]                     
__________________________________________________________________________________________________
res2c_branch2a (Conv2D)         (None, 15, 15, 64)   16448       activation_55[0][0]              
__________________________________________________________________________________________________
bn2c_branch2a (BatchNormalizati (None, 15, 15, 64)   256         res2c_branch2a[0][0]             
__________________________________________________________________________________________________
activation_56 (Activation)      (None, 15, 15, 64)   0           bn2c_branch2a[0][0]              
__________________________________________________________________________________________________
res2c_branch2b (Conv2D)         (None, 15, 15, 64)   36928       activation_56[0][0]              
__________________________________________________________________________________________________
bn2c_branch2b (BatchNormalizati (None, 15, 15, 64)   256         res2c_branch2b[0][0]             
__________________________________________________________________________________________________
activation_57 (Activation)      (None, 15, 15, 64)   0           bn2c_branch2b[0][0]              
__________________________________________________________________________________________________
res2c_branch2c (Conv2D)         (None, 15, 15, 256)  16640       activation_57[0][0]              
__________________________________________________________________________________________________
bn2c_branch2c (BatchNormalizati (None, 15, 15, 256)  1024        res2c_branch2c[0][0]             
__________________________________________________________________________________________________
add_18 (Add)                    (None, 15, 15, 256)  0           activation_55[0][0]              
                                                                 bn2c_branch2c[0][0]              
__________________________________________________________________________________________________
activation_58 (Activation)      (None, 15, 15, 256)  0           add_18[0][0]                     
__________________________________________________________________________________________________
res3a_branch2a (Conv2D)         (None, 8, 8, 128)    32896       activation_58[0][0]              
__________________________________________________________________________________________________
bn3a_branch2a (BatchNormalizati (None, 8, 8, 128)    512         res3a_branch2a[0][0]             
__________________________________________________________________________________________________
activation_59 (Activation)      (None, 8, 8, 128)    0           bn3a_branch2a[0][0]              
__________________________________________________________________________________________________
res3a_branch2b (Conv2D)         (None, 8, 8, 128)    147584      activation_59[0][0]              
__________________________________________________________________________________________________
bn3a_branch2b (BatchNormalizati (None, 8, 8, 128)    512         res3a_branch2b[0][0]             
__________________________________________________________________________________________________
activation_60 (Activation)      (None, 8, 8, 128)    0           bn3a_branch2b[0][0]              
__________________________________________________________________________________________________
res3a_branch1 (Conv2D)          (None, 8, 8, 512)    131584      activation_58[0][0]              
__________________________________________________________________________________________________
res3a_branch2c (Conv2D)         (None, 8, 8, 512)    66048       activation_60[0][0]              
__________________________________________________________________________________________________
bn3a_branch1 (BatchNormalizatio (None, 8, 8, 512)    2048        res3a_branch1[0][0]              
__________________________________________________________________________________________________
bn3a_branch2c (BatchNormalizati (None, 8, 8, 512)    2048        res3a_branch2c[0][0]             
__________________________________________________________________________________________________
add_19 (Add)                    (None, 8, 8, 512)    0           bn3a_branch1[0][0]               
                                                                 bn3a_branch2c[0][0]              
__________________________________________________________________________________________________
activation_61 (Activation)      (None, 8, 8, 512)    0           add_19[0][0]                     
__________________________________________________________________________________________________
res3b_branch2a (Conv2D)         (None, 8, 8, 128)    65664       activation_61[0][0]              
__________________________________________________________________________________________________
bn3b_branch2a (BatchNormalizati (None, 8, 8, 128)    512         res3b_branch2a[0][0]             
__________________________________________________________________________________________________
activation_62 (Activation)      (None, 8, 8, 128)    0           bn3b_branch2a[0][0]              
__________________________________________________________________________________________________
res3b_branch2b (Conv2D)         (None, 8, 8, 128)    147584      activation_62[0][0]              
__________________________________________________________________________________________________
bn3b_branch2b (BatchNormalizati (None, 8, 8, 128)    512         res3b_branch2b[0][0]             
__________________________________________________________________________________________________
activation_63 (Activation)      (None, 8, 8, 128)    0           bn3b_branch2b[0][0]              
__________________________________________________________________________________________________
res3b_branch2c (Conv2D)         (None, 8, 8, 512)    66048       activation_63[0][0]              
__________________________________________________________________________________________________
bn3b_branch2c (BatchNormalizati (None, 8, 8, 512)    2048        res3b_branch2c[0][0]             
__________________________________________________________________________________________________
add_20 (Add)                    (None, 8, 8, 512)    0           activation_61[0][0]              
                                                                 bn3b_branch2c[0][0]              
__________________________________________________________________________________________________
activation_64 (Activation)      (None, 8, 8, 512)    0           add_20[0][0]                     
__________________________________________________________________________________________________
res3c_branch2a (Conv2D)         (None, 8, 8, 128)    65664       activation_64[0][0]              
__________________________________________________________________________________________________
bn3c_branch2a (BatchNormalizati (None, 8, 8, 128)    512         res3c_branch2a[0][0]             
__________________________________________________________________________________________________
activation_65 (Activation)      (None, 8, 8, 128)    0           bn3c_branch2a[0][0]              
__________________________________________________________________________________________________
res3c_branch2b (Conv2D)         (None, 8, 8, 128)    147584      activation_65[0][0]              
__________________________________________________________________________________________________
bn3c_branch2b (BatchNormalizati (None, 8, 8, 128)    512         res3c_branch2b[0][0]             
__________________________________________________________________________________________________
activation_66 (Activation)      (None, 8, 8, 128)    0           bn3c_branch2b[0][0]              
__________________________________________________________________________________________________
res3c_branch2c (Conv2D)         (None, 8, 8, 512)    66048       activation_66[0][0]              
__________________________________________________________________________________________________
bn3c_branch2c (BatchNormalizati (None, 8, 8, 512)    2048        res3c_branch2c[0][0]             
__________________________________________________________________________________________________
add_21 (Add)                    (None, 8, 8, 512)    0           activation_64[0][0]              
                                                                 bn3c_branch2c[0][0]              
__________________________________________________________________________________________________
activation_67 (Activation)      (None, 8, 8, 512)    0           add_21[0][0]                     
__________________________________________________________________________________________________
res3d_branch2a (Conv2D)         (None, 8, 8, 128)    65664       activation_67[0][0]              
__________________________________________________________________________________________________
bn3d_branch2a (BatchNormalizati (None, 8, 8, 128)    512         res3d_branch2a[0][0]             
__________________________________________________________________________________________________
activation_68 (Activation)      (None, 8, 8, 128)    0           bn3d_branch2a[0][0]              
__________________________________________________________________________________________________
res3d_branch2b (Conv2D)         (None, 8, 8, 128)    147584      activation_68[0][0]              
__________________________________________________________________________________________________
bn3d_branch2b (BatchNormalizati (None, 8, 8, 128)    512         res3d_branch2b[0][0]             
__________________________________________________________________________________________________
activation_69 (Activation)      (None, 8, 8, 128)    0           bn3d_branch2b[0][0]              
__________________________________________________________________________________________________
res3d_branch2c (Conv2D)         (None, 8, 8, 512)    66048       activation_69[0][0]              
__________________________________________________________________________________________________
bn3d_branch2c (BatchNormalizati (None, 8, 8, 512)    2048        res3d_branch2c[0][0]             
__________________________________________________________________________________________________
add_22 (Add)                    (None, 8, 8, 512)    0           activation_67[0][0]              
                                                                 bn3d_branch2c[0][0]              
__________________________________________________________________________________________________
activation_70 (Activation)      (None, 8, 8, 512)    0           add_22[0][0]                     
__________________________________________________________________________________________________
res4a_branch2a (Conv2D)         (None, 4, 4, 256)    131328      activation_70[0][0]              
__________________________________________________________________________________________________
bn4a_branch2a (BatchNormalizati (None, 4, 4, 256)    1024        res4a_branch2a[0][0]             
__________________________________________________________________________________________________
activation_71 (Activation)      (None, 4, 4, 256)    0           bn4a_branch2a[0][0]              
__________________________________________________________________________________________________
res4a_branch2b (Conv2D)         (None, 4, 4, 256)    590080      activation_71[0][0]              
__________________________________________________________________________________________________
bn4a_branch2b (BatchNormalizati (None, 4, 4, 256)    1024        res4a_branch2b[0][0]             
__________________________________________________________________________________________________
activation_72 (Activation)      (None, 4, 4, 256)    0           bn4a_branch2b[0][0]              
__________________________________________________________________________________________________
res4a_branch1 (Conv2D)          (None, 4, 4, 1024)   525312      activation_70[0][0]              
__________________________________________________________________________________________________
res4a_branch2c (Conv2D)         (None, 4, 4, 1024)   263168      activation_72[0][0]              
__________________________________________________________________________________________________
bn4a_branch1 (BatchNormalizatio (None, 4, 4, 1024)   4096        res4a_branch1[0][0]              
__________________________________________________________________________________________________
bn4a_branch2c (BatchNormalizati (None, 4, 4, 1024)   4096        res4a_branch2c[0][0]             
__________________________________________________________________________________________________
add_23 (Add)                    (None, 4, 4, 1024)   0           bn4a_branch1[0][0]               
                                                                 bn4a_branch2c[0][0]              
__________________________________________________________________________________________________
activation_73 (Activation)      (None, 4, 4, 1024)   0           add_23[0][0]                     
__________________________________________________________________________________________________
res4b_branch2a (Conv2D)         (None, 4, 4, 256)    262400      activation_73[0][0]              
__________________________________________________________________________________________________
bn4b_branch2a (BatchNormalizati (None, 4, 4, 256)    1024        res4b_branch2a[0][0]             
__________________________________________________________________________________________________
activation_74 (Activation)      (None, 4, 4, 256)    0           bn4b_branch2a[0][0]              
__________________________________________________________________________________________________
res4b_branch2b (Conv2D)         (None, 4, 4, 256)    590080      activation_74[0][0]              
__________________________________________________________________________________________________
bn4b_branch2b (BatchNormalizati (None, 4, 4, 256)    1024        res4b_branch2b[0][0]             
__________________________________________________________________________________________________
activation_75 (Activation)      (None, 4, 4, 256)    0           bn4b_branch2b[0][0]              
__________________________________________________________________________________________________
res4b_branch2c (Conv2D)         (None, 4, 4, 1024)   263168      activation_75[0][0]              
__________________________________________________________________________________________________
bn4b_branch2c (BatchNormalizati (None, 4, 4, 1024)   4096        res4b_branch2c[0][0]             
__________________________________________________________________________________________________
add_24 (Add)                    (None, 4, 4, 1024)   0           activation_73[0][0]              
                                                                 bn4b_branch2c[0][0]              
__________________________________________________________________________________________________
activation_76 (Activation)      (None, 4, 4, 1024)   0           add_24[0][0]                     
__________________________________________________________________________________________________
res4c_branch2a (Conv2D)         (None, 4, 4, 256)    262400      activation_76[0][0]              
__________________________________________________________________________________________________
bn4c_branch2a (BatchNormalizati (None, 4, 4, 256)    1024        res4c_branch2a[0][0]             
__________________________________________________________________________________________________
activation_77 (Activation)      (None, 4, 4, 256)    0           bn4c_branch2a[0][0]              
__________________________________________________________________________________________________
res4c_branch2b (Conv2D)         (None, 4, 4, 256)    590080      activation_77[0][0]              
__________________________________________________________________________________________________
bn4c_branch2b (BatchNormalizati (None, 4, 4, 256)    1024        res4c_branch2b[0][0]             
__________________________________________________________________________________________________
activation_78 (Activation)      (None, 4, 4, 256)    0           bn4c_branch2b[0][0]              
__________________________________________________________________________________________________
res4c_branch2c (Conv2D)         (None, 4, 4, 1024)   263168      activation_78[0][0]              
__________________________________________________________________________________________________
bn4c_branch2c (BatchNormalizati (None, 4, 4, 1024)   4096        res4c_branch2c[0][0]             
__________________________________________________________________________________________________
add_25 (Add)                    (None, 4, 4, 1024)   0           activation_76[0][0]              
                                                                 bn4c_branch2c[0][0]              
__________________________________________________________________________________________________
activation_79 (Activation)      (None, 4, 4, 1024)   0           add_25[0][0]                     
__________________________________________________________________________________________________
res4d_branch2a (Conv2D)         (None, 4, 4, 256)    262400      activation_79[0][0]              
__________________________________________________________________________________________________
bn4d_branch2a (BatchNormalizati (None, 4, 4, 256)    1024        res4d_branch2a[0][0]             
__________________________________________________________________________________________________
activation_80 (Activation)      (None, 4, 4, 256)    0           bn4d_branch2a[0][0]              
__________________________________________________________________________________________________
res4d_branch2b (Conv2D)         (None, 4, 4, 256)    590080      activation_80[0][0]              
__________________________________________________________________________________________________
bn4d_branch2b (BatchNormalizati (None, 4, 4, 256)    1024        res4d_branch2b[0][0]             
__________________________________________________________________________________________________
activation_81 (Activation)      (None, 4, 4, 256)    0           bn4d_branch2b[0][0]              
__________________________________________________________________________________________________
res4d_branch2c (Conv2D)         (None, 4, 4, 1024)   263168      activation_81[0][0]              
__________________________________________________________________________________________________
bn4d_branch2c (BatchNormalizati (None, 4, 4, 1024)   4096        res4d_branch2c[0][0]             
__________________________________________________________________________________________________
add_26 (Add)                    (None, 4, 4, 1024)   0           activation_79[0][0]              
                                                                 bn4d_branch2c[0][0]              
__________________________________________________________________________________________________
activation_82 (Activation)      (None, 4, 4, 1024)   0           add_26[0][0]                     
__________________________________________________________________________________________________
res4e_branch2a (Conv2D)         (None, 4, 4, 256)    262400      activation_82[0][0]              
__________________________________________________________________________________________________
bn4e_branch2a (BatchNormalizati (None, 4, 4, 256)    1024        res4e_branch2a[0][0]             
__________________________________________________________________________________________________
activation_83 (Activation)      (None, 4, 4, 256)    0           bn4e_branch2a[0][0]              
__________________________________________________________________________________________________
res4e_branch2b (Conv2D)         (None, 4, 4, 256)    590080      activation_83[0][0]              
__________________________________________________________________________________________________
bn4e_branch2b (BatchNormalizati (None, 4, 4, 256)    1024        res4e_branch2b[0][0]             
__________________________________________________________________________________________________
activation_84 (Activation)      (None, 4, 4, 256)    0           bn4e_branch2b[0][0]              
__________________________________________________________________________________________________
res4e_branch2c (Conv2D)         (None, 4, 4, 1024)   263168      activation_84[0][0]              
__________________________________________________________________________________________________
bn4e_branch2c (BatchNormalizati (None, 4, 4, 1024)   4096        res4e_branch2c[0][0]             
__________________________________________________________________________________________________
add_27 (Add)                    (None, 4, 4, 1024)   0           activation_82[0][0]              
                                                                 bn4e_branch2c[0][0]              
__________________________________________________________________________________________________
activation_85 (Activation)      (None, 4, 4, 1024)   0           add_27[0][0]                     
__________________________________________________________________________________________________
res4f_branch2a (Conv2D)         (None, 4, 4, 256)    262400      activation_85[0][0]              
__________________________________________________________________________________________________
bn4f_branch2a (BatchNormalizati (None, 4, 4, 256)    1024        res4f_branch2a[0][0]             
__________________________________________________________________________________________________
activation_86 (Activation)      (None, 4, 4, 256)    0           bn4f_branch2a[0][0]              
__________________________________________________________________________________________________
res4f_branch2b (Conv2D)         (None, 4, 4, 256)    590080      activation_86[0][0]              
__________________________________________________________________________________________________
bn4f_branch2b (BatchNormalizati (None, 4, 4, 256)    1024        res4f_branch2b[0][0]             
__________________________________________________________________________________________________
activation_87 (Activation)      (None, 4, 4, 256)    0           bn4f_branch2b[0][0]              
__________________________________________________________________________________________________
res4f_branch2c (Conv2D)         (None, 4, 4, 1024)   263168      activation_87[0][0]              
__________________________________________________________________________________________________
bn4f_branch2c (BatchNormalizati (None, 4, 4, 1024)   4096        res4f_branch2c[0][0]             
__________________________________________________________________________________________________
add_28 (Add)                    (None, 4, 4, 1024)   0           activation_85[0][0]              
                                                                 bn4f_branch2c[0][0]              
__________________________________________________________________________________________________
activation_88 (Activation)      (None, 4, 4, 1024)   0           add_28[0][0]                     
__________________________________________________________________________________________________
res5a_branch2a (Conv2D)         (None, 2, 2, 512)    524800      activation_88[0][0]              
__________________________________________________________________________________________________
bn5a_branch2a (BatchNormalizati (None, 2, 2, 512)    2048        res5a_branch2a[0][0]             
__________________________________________________________________________________________________
activation_89 (Activation)      (None, 2, 2, 512)    0           bn5a_branch2a[0][0]              
__________________________________________________________________________________________________
res5a_branch2b (Conv2D)         (None, 2, 2, 512)    2359808     activation_89[0][0]              
__________________________________________________________________________________________________
bn5a_branch2b (BatchNormalizati (None, 2, 2, 512)    2048        res5a_branch2b[0][0]             
__________________________________________________________________________________________________
activation_90 (Activation)      (None, 2, 2, 512)    0           bn5a_branch2b[0][0]              
__________________________________________________________________________________________________
res5a_branch1 (Conv2D)          (None, 2, 2, 2048)   2099200     activation_88[0][0]              
__________________________________________________________________________________________________
res5a_branch2c (Conv2D)         (None, 2, 2, 2048)   1050624     activation_90[0][0]              
__________________________________________________________________________________________________
bn5a_branch1 (BatchNormalizatio (None, 2, 2, 2048)   8192        res5a_branch1[0][0]              
__________________________________________________________________________________________________
bn5a_branch2c (BatchNormalizati (None, 2, 2, 2048)   8192        res5a_branch2c[0][0]             
__________________________________________________________________________________________________
add_29 (Add)                    (None, 2, 2, 2048)   0           bn5a_branch1[0][0]               
                                                                 bn5a_branch2c[0][0]              
__________________________________________________________________________________________________
activation_91 (Activation)      (None, 2, 2, 2048)   0           add_29[0][0]                     
__________________________________________________________________________________________________
res5b_branch2a (Conv2D)         (None, 2, 2, 512)    1049088     activation_91[0][0]              
__________________________________________________________________________________________________
bn5b_branch2a (BatchNormalizati (None, 2, 2, 512)    2048        res5b_branch2a[0][0]             
__________________________________________________________________________________________________
activation_92 (Activation)      (None, 2, 2, 512)    0           bn5b_branch2a[0][0]              
__________________________________________________________________________________________________
res5b_branch2b (Conv2D)         (None, 2, 2, 512)    2359808     activation_92[0][0]              
__________________________________________________________________________________________________
bn5b_branch2b (BatchNormalizati (None, 2, 2, 512)    2048        res5b_branch2b[0][0]             
__________________________________________________________________________________________________
activation_93 (Activation)      (None, 2, 2, 512)    0           bn5b_branch2b[0][0]              
__________________________________________________________________________________________________
res5b_branch2c (Conv2D)         (None, 2, 2, 2048)   1050624     activation_93[0][0]              
__________________________________________________________________________________________________
bn5b_branch2c (BatchNormalizati (None, 2, 2, 2048)   8192        res5b_branch2c[0][0]             
__________________________________________________________________________________________________
add_30 (Add)                    (None, 2, 2, 2048)   0           activation_91[0][0]              
                                                                 bn5b_branch2c[0][0]              
__________________________________________________________________________________________________
activation_94 (Activation)      (None, 2, 2, 2048)   0           add_30[0][0]                     
__________________________________________________________________________________________________
res5c_branch2a (Conv2D)         (None, 2, 2, 512)    1049088     activation_94[0][0]              
__________________________________________________________________________________________________
bn5c_branch2a (BatchNormalizati (None, 2, 2, 512)    2048        res5c_branch2a[0][0]             
__________________________________________________________________________________________________
activation_95 (Activation)      (None, 2, 2, 512)    0           bn5c_branch2a[0][0]              
__________________________________________________________________________________________________
res5c_branch2b (Conv2D)         (None, 2, 2, 512)    2359808     activation_95[0][0]              
__________________________________________________________________________________________________
bn5c_branch2b (BatchNormalizati (None, 2, 2, 512)    2048        res5c_branch2b[0][0]             
__________________________________________________________________________________________________
activation_96 (Activation)      (None, 2, 2, 512)    0           bn5c_branch2b[0][0]              
__________________________________________________________________________________________________
res5c_branch2c (Conv2D)         (None, 2, 2, 2048)   1050624     activation_96[0][0]              
__________________________________________________________________________________________________
bn5c_branch2c (BatchNormalizati (None, 2, 2, 2048)   8192        res5c_branch2c[0][0]             
__________________________________________________________________________________________________
add_31 (Add)                    (None, 2, 2, 2048)   0           activation_94[0][0]              
                                                                 bn5c_branch2c[0][0]              
__________________________________________________________________________________________________
activation_97 (Activation)      (None, 2, 2, 2048)   0           add_31[0][0]                     
__________________________________________________________________________________________________
avg_pool (AveragePooling2D)     (None, 1, 1, 2048)   0           activation_97[0][0]              
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 2048)         0           avg_pool[0][0]                   
__________________________________________________________________________________________________
fc6 (Dense)                     (None, 6)            12294       flatten_1[0][0]                  
==================================================================================================
Total params: 23,600,006
Trainable params: 23,546,886
Non-trainable params: 53,120
__________________________________________________________________________________________________
