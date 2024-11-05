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
    Input, Conv2D, BatchNormalization, Dropout, UpSampling2D, GlobalAveragePooling2D, Reshape, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

# Custom layer for grayscale to RGB conversion
class GrayscaleToRGB(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.image.grayscale_to_rgb(inputs)

# ASKCFuse layer for adaptive attention-guided fusion
class ASKCFuse(tf.keras.layers.Layer):
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
        xa = tf.keras.layers.Add()([x, residual])

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
        xlg = tf.keras.layers.Add()([xl, xg])
        wei = self.sig(xlg)

        # Final output with scaled attention
        fused_output = tf.keras.layers.Add()([
            tf.keras.layers.Multiply()([x, 2 * wei]),
            tf.keras.layers.Multiply()([residual, 2 * (1 - wei)])
        ])
        return fused_output
# QualityNet for adaptive attention between RegNet and DetNet outputs
class QualityNet(tf.keras.layers.Layer):
    def __init__(self):
        super(QualityNet, self).__init__()
        self.resize_layer = tf.keras.layers.Resizing(64, 80)  # Resize image_input to match reg_output/det_output
        self.conv1 = Conv2D(64, (7, 7), padding='same', activation='relu')
        self.conv2 = Conv2D(32, (5, 5), padding='same', activation='relu')
        self.conv3 = Conv2D(16, (3, 3), padding='same', activation='relu')
        self.conv4 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')  # Produces attention map

    def call(self, reg_output, det_output, image_input):
        resized_image_input = self.resize_layer(image_input)  # Resize image to match feature map dimensions
        x = tf.concat([reg_output, det_output, resized_image_input], axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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
    fused_output = tf.keras.layers.Add()([
        tf.keras.layers.Multiply()([attention_map, det_conv]),
        tf.keras.layers.Multiply()([(1 - attention_map), reg_conv])
    ])
    final_output = Conv2D(1, (1, 1), padding='same', activation='relu')(fused_output)

    return Model(inputs=[inputs_image, inputs_detection], outputs=final_output)

# Create the model
model = build_decidenet_with_attention()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np  # Ensure you have this import if using numpy arrays

# Define a data generator (assuming data in NumPy format or other preprocessable format)
class CrowdCountingDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, detection_maps, density_maps, batch_size=8):
        self.images = images
        self.detection_maps = detection_maps
        self.density_maps = density_maps
        self.batch_size = batch_size
        self.indices = np.arange(len(self.images))

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return self.images[batch_indices], self.density_maps[batch_indices]  # Adjust as necessary for detection maps

# Initialize data generators
train_generator = CrowdCountingDataGenerator(x_train, None, y_train, batch_size=8)  # Assuming None for detection maps
val_generator = CrowdCountingDataGenerator(x_val, None, y_val, batch_size=8)  # Assuming None for detection maps

# Model setup
def build_decidenet_with_attention():
    # Define your model architecture here
    # Example using ResNet50 as a backbone
    base_model = ResNet50(input_shape=(240, 320, 1), include_top=False, weights=None)
    x = base_model.output
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1, (1, 1), activation='linear')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

model = build_decidenet_with_attention()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Define model checkpoint to save the best model during training
checkpoint_path = "decidenet_best_model.keras"
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)

# Training the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,  # Adjust the number of epochs as needed
    callbacks=[checkpoint]
)

# Save the final model
model.save("decidenet_final_model.keras")
print("Model saved successfully.")
