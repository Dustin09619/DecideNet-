# Install necessary libraries
!pip install tensorflow pandas
import os
import numpy as np
import cv2
import scipy.io as sio
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Concatenate, Add, AveragePooling2D, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.applications import ResNet50

# Enable mixed precision for memory optimization
from tensorflow.keras.mixed_precision import Policy
policy = Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Define paths
dataset_path = '/content/drive/MyDrive/mall_dataset'
frames_path = os.path.join(dataset_path, 'resized_images')
gt_file = os.path.join(dataset_path, 'mall_gt_with_density.mat')

# Load and Prepare Data with resized images
def data_preparation():
    print('Loading data...')
    frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith('.jpg')])
    images = [cv2.imread(os.path.join(frames_path, file), cv2.IMREAD_GRAYSCALE) for file in frame_files]

    # Resize images to 320x240 and normalize
    images = [cv2.resize(img, (320, 240)) for img in images]
    images = [(img - 127.5) / 128 for img in images]
    images = np.array([np.expand_dims(img, axis=-1) for img in images])

    # Load ground-truth annotations and resize to 320x240
    gt_data = sio.loadmat(gt_file)
    density_maps = gt_data['density_map']

    # Check if density_maps is correctly shaped and resize accordingly
    if density_maps.ndim == 2:  # If density_maps is in 2D, adjust this logic as necessary
        density_maps = np.expand_dims(density_maps, axis=-1)
    density_maps_resized = [cv2.resize(dm, (320, 240)) for dm in density_maps]

    print('Data loaded.')
    return images, np.array(density_maps_resized)

# Load data
x_data, y_data = data_preparation()

# Split data (800 training, 100 validation from training, 1200 testing)
np.random.seed(42)
x_train = x_data[:800]
y_train = y_data[:800]

indices = np.random.choice(range(800), 100, replace=False)
x_val = x_train[indices]
y_val = y_train[indices]

x_test = x_data[800:]
y_test = y_data[800:]


import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Dropout, UpSampling2D, GlobalAveragePooling2D, Reshape, Activation,
    Multiply, Add, Layer, Resizing
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50

# Custom layer for grayscale to RGB conversion
class GrayscaleToRGB(Layer):
    def call(self, inputs):
        return tf.image.grayscale_to_rgb(inputs)

# ASKCFuse layer for adaptive attention-guided fusion
class ASKCFuse(Layer):
    def __init__(self, channels=64, r=4):
        super(ASKCFuse, self).__init__()
        inter_channels = channels // r

        # Local Attention Layers
        self.local_att_conv1 = Conv2D(inter_channels, kernel_size=1, padding='same')
        self.local_att_bn1 = BatchNormalization()
        self.local_att_act1 = Activation('relu')
        self.local_att_conv2 = Conv2D(channels, kernel_size=1, padding='same')
        self.local_att_bn2 = BatchNormalization()

        # Global Attention Layers
        self.global_att_pool = GlobalAveragePooling2D()
        self.global_att_conv1 = Conv2D(inter_channels, kernel_size=1, padding='same')
        self.global_att_bn1 = BatchNormalization()
        self.global_att_act1 = Activation('relu')
        self.global_att_conv2 = Conv2D(channels, kernel_size=1, padding='same')
        self.global_att_bn2 = BatchNormalization()

        self.sig = Activation('sigmoid')

    def call(self, x, residual):
        xa = Add()([x, residual])

        # Local Attention
        xl = self.local_att_conv1(xa)
        xl = self.local_att_bn1(xl)
        xl = self.local_att_act1(xl)
        xl = self.local_att_conv2(xl)
        xl = self.local_att_bn2(xl)

        # Global Attention
        xg = self.global_att_pool(xa)
        xg = Reshape((1, 1, -1))(xg)
        xg = self.global_att_conv1(xg)
        xg = self.global_att_bn1(xg)
        xg = self.global_att_act1(xg)
        xg = self.global_att_conv2(xg)
        xg = self.global_att_bn2(xg)

        # Combine Local and Global Attention
        xlg = Add()([xl, xg])
        wei = self.sig(xlg)

        # Final output with scaled attention
        fused_output = Add()([
            Multiply()([x, 2 * wei]),
            Multiply()([residual, 2 * (1 - wei)])
        ])
        return fused_output

class QualityNet(Layer):
    def __init__(self):
        super(QualityNet, self).__init__()
        self.conv1 = Conv2D(64, (7, 7), padding='same', activation='relu')
        self.conv2 = Conv2D(32, (5, 5), padding='same', activation='relu')
        self.conv3 = Conv2D(16, (3, 3), padding='same', activation='relu')
        self.conv4 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')  # Produces attention map
        self.resize_layer = Resizing(64, 80)

    def call(self, reg_output, det_output, image_input):
        # Resize image_input to match the dimensions of reg_output and det_output
        resized_image_input = self.resize_layer(image_input)
        
        # Concatenate along the channel dimension
        x = tf.concat([reg_output, det_output, resized_image_input], axis=-1)
        
        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Generate the attention map
        attention_map = self.conv4(x)
        
        return attention_map

def build_decidenet_with_attention():
    inputs_image = Input(shape=(240, 320, 1), name='inputs_image')
    inputs_detection = Input(shape=(240, 320, 1), name='inputs_detection')

    # Convert grayscale input to RGB
    rgb_image = GrayscaleToRGB()(inputs_image)
    rgb_detection = GrayscaleToRGB()(inputs_detection)

    # Use Keras's built-in ResNet50
    backbone = ResNet50(weights='imagenet', include_top=False, input_shape=(240, 320, 3))
    for layer in backbone.layers:
        layer.trainable = False  # Freeze the backbone layers

    # Separate paths for DetNet and RegNet
    detnet_features = backbone(rgb_detection)
    detnet_features = UpSampling2D(size=(8, 8))(detnet_features)
    det_conv = Conv2D(48, (5, 5), padding='same', activation='relu')(detnet_features)
    det_conv = BatchNormalization()(det_conv)
    det_conv = Dropout(0.3)(det_conv)
    det_conv = Conv2D(1, (1, 1), padding='same')(det_conv)

    regnet_features = backbone(rgb_image)
    reg_conv = UpSampling2D(size=(8, 8))(regnet_features)
    reg_conv = Conv2D(32, (5, 5), padding='same', activation='relu')(reg_conv)
    reg_conv = BatchNormalization()(reg_conv)
    reg_conv = Dropout(0.3)(reg_conv)
    reg_conv = Conv2D(1, (1, 1), padding='same')(reg_conv)

    # QualityNet for adaptive attention between RegNet and DetNet outputs
    quality_net = QualityNet()
    attention_map = quality_net(reg_conv, det_conv, rgb_image)

    # Attention fusion
    fused_output = Add()([
        Multiply()([attention_map, det_conv]),
        Multiply()([(1 - attention_map), reg_conv])
    ])

    # Upsample fused_output to match the target shape of (240, 320, 1)
    fused_output = UpSampling2D(size=(4, 4))(fused_output)  # Adjust the upsampling factor if necessary
    final_output = Conv2D(1, (1, 1), padding='same', activation='linear')(fused_output)
    final_output = Resizing(240, 320)(final_output)

    model = Model(inputs=[inputs_image, inputs_detection], outputs=[final_output])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])

    return model

# Build the model
model = build_decidenet_with_attention()

# Optionally, print the model summary to verify its structure
model.summary()
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Define callbacks
callbacks = [
    ModelCheckpoint(
        filepath='decidenet_model.keras',  # Save the model in Keras format
        monitor='val_loss',  # Monitor the validation loss
        save_best_only=True,  # Save only the best model
        mode='min',  # We want to minimize the loss
        verbose=1  # Verbose output for better visibility
    ),
    EarlyStopping(
        monitor='val_loss',  # Stop training when the validation loss has stopped improving
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        verbose=1  # Verbose output
    ),
    ReduceLROnPlateau(
        monitor='val_loss',  # Reduce learning rate when the validation loss has stopped improving
        factor=0.1,  # Factor by which the learning rate will be reduced
        patience=5,  # Number of epochs with no improvement after which learning rate will be reduced
        verbose=1,  # Verbose output
        min_lr=1e-6  # Lower bound on the learning rate
    )
]
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generator_two_inputs(x, y):
    data_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],  # Brightness augmentation
        shear_range=0.2  # Shear transformation
    )

    while True:
        for gen_x, gen_y in data_gen.flow(x, y, batch_size=16):
            # Check the shape of gen_y
            if len(gen_y.shape) == 3:  # (batch_size, height, width)
                gen_y = np.expand_dims(gen_y, axis=-1)  # Add a channel dimension
            
            # Yielding the data
            yield {"inputs_image": gen_x, "inputs_detection": gen_x}, gen_y

# Create a TensorFlow Dataset from the generator
train_dataset = tf.data.Dataset.from_generator(
    lambda: generator_two_inputs(x_train, y_train),
    output_signature=(
        {
            "inputs_image": tf.TensorSpec(shape=(None, 240, 320, 1), dtype=tf.float32),
            "inputs_detection": tf.TensorSpec(shape=(None, 240, 320, 1), dtype=tf.float32)
        },
        tf.TensorSpec(shape=(None, 240, 320, 1), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)

# Fit the model using the custom dataset
history = model.fit(
    train_dataset,
    steps_per_epoch=len(x_train) // 16,
    epochs=150,
    validation_data=({"inputs_image": x_val, "inputs_detection": x_val}, np.expand_dims(y_val, axis=-1)),
    callbacks=callbacks  # Use the correct variable name here
)

# Save the final model
model.save('DecideNet_mall_optimized.h5')

