!pip install tensorflow

import os
import numpy as np
import cv2
import scipy.io as sio
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Add, Activation, AveragePooling2D, Resizing
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Define paths
dataset_path = '/content/drive/MyDrive/mall_dataset'
frames_path = os.path.join(dataset_path, 'frames')
feat_file = os.path.join(dataset_path, 'mall_feat.mat')
gt_file = os.path.join(dataset_path, 'mall_gt_with_density.mat')

# Verify files
image_files = [os.path.join(frames_path, f) for f in os.listdir(frames_path) if f.endswith('.jpg')]
print("First few image files:", image_files[:5])
print("Features file exists:", os.path.exists(feat_file))
print("Ground truth file exists:", os.path.exists(gt_file))

# Load and Prepare Data
def data_preparation():
    print('Loading data...')
    frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith('.jpg')])
    images = [cv2.imread(os.path.join(frames_path, file), cv2.IMREAD_GRAYSCALE) for file in frame_files]
    images = [(img - 127.5) / 128 for img in images]  # Normalize images
    images = np.array([np.expand_dims(img, axis=-1) for img in images])
    
    # Load ground-truth annotations
    gt_data = sio.loadmat(gt_file)
    print("Keys in ground truth data:", gt_data.keys())

    density_maps = gt_data['density_map']  # Assuming density maps are stored in 'density_map'
    print("Shape of density maps:", density_maps.shape)  # Verify shape of density maps
    
    print('Data loaded.')
    return images, density_maps

# Load data
x_data, y_data = data_preparation()

# Split data (800 training, 100 validation from training, 1200 testing)
np.random.seed(42)  # Set seed for reproducibility
x_train = x_data[:800]
y_train = y_data[:800]

# Randomly select 100 images for validation from the training set
indices = np.random.choice(range(800), 100, replace=False)
x_val = x_train[indices]
y_val = y_train[indices]

# Remaining 1200 images are for testing
x_test = x_data[800:]
y_test = y_data[800:]

# Data Augmentation
data_gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)


# Define custom loss functions
import tensorflow as tf

def maaae(y_true, y_pred):
    s = tf.reduce_sum(tf.reduce_sum(y_true, axis=1), axis=1)
    s1 = tf.reduce_sum(tf.reduce_sum(y_pred, axis=1), axis=1)
    return tf.reduce_mean(tf.abs(s - s1))

def mssse(y_true, y_pred):
    s = tf.reduce_sum(tf.reduce_sum(y_true, axis=1), axis=1)
    s1 = tf.reduce_sum(tf.reduce_sum(y_pred, axis=1), axis=1)
    return tf.reduce_mean(tf.square(s - s1))

def customLoss(y_true, y_pred):
    # Resize y_true to match y_pred dimensions, explicitly specifying channels
    y_true = tf.image.resize(y_true[..., tf.newaxis], [tf.shape(y_pred)[1], tf.shape(y_pred)[2]], method='bilinear')
    y_true = tf.squeeze(y_true, axis=-1) # Remove channel dimension

    loss1 = mssse(y_true, y_pred) 
    loss2 = tf.reduce_mean(tf.square(y_true - y_pred))  
    return 0.7 * loss1 + 0.3 * loss2

def build_decidenet():
    # Inputs
    inputs_image = Input(shape=(480, 640, 1), name='inputs_image')
    inputs_detection = Input(shape=(480, 640, 1), name='inputs_detection')

    # Regression-based density estimation (RegNet)
    reg_conv = Conv2D(24, (5, 5), padding='same', activation='relu')(inputs_image)
    reg_conv = MaxPooling2D(pool_size=(2, 2))(reg_conv)  # Output shape: (240, 320, 24)
    reg_conv = Conv2D(48, (3, 3), padding='same', activation='relu')(reg_conv)
    reg_conv = Conv2D(1, (1, 1), padding='same')(reg_conv)  # Output shape: (240, 320, 1)

    # Detection-based density estimation (DetNet)
    det_conv = Conv2D(16, (7, 7), padding='same', activation='relu')(inputs_detection)
    det_conv = MaxPooling2D(pool_size=(2, 2))(det_conv)  # Output shape: (240, 320, 16)
    det_conv = Conv2D(32, (5, 5), padding='same', activation='relu')(det_conv)
    det_conv = Conv2D(1, (1, 1), padding='same')(det_conv)  # Output shape: (240, 320, 1)

    # Downsample the input image to match the shape of reg_conv and det_conv
    downsampled_input_image = AveragePooling2D(pool_size=(2, 2))(inputs_image)  # Output shape: (240, 320, 1)

    # Concatenate with downsampled input
    attention_input = Concatenate(axis=3)([reg_conv, det_conv, downsampled_input_image])
    attention_conv = Conv2D(16, (3, 3), padding='same', activation='relu')(attention_input)
    attention_conv = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(attention_conv)

    # Combine results based on attention
    final_output = Add()([reg_conv * (1 - attention_conv), det_conv * attention_conv])


    # Create the model
    return Model(inputs=[inputs_image, inputs_detection], outputs=final_output)

# Build and compile the model
model = build_decidenet()
adam = Adam(learning_rate=5e-3)
model.compile(loss=customLoss, optimizer=adam, metrics=[maaae, mssse])

# Training setup
reduce_lr = ReduceLROnPlateau(monitor='val_maaae', factor=0.90, patience=10, min_lr=1e-5)
tensorboard = TensorBoard(log_dir='./logs/DecideNet', write_graph=True)
callbacks_list = [reduce_lr, tensorboard]

# Updated generator function to yield two inputs and one output
def generator_two_inputs(data_gen, x, y):
    while True:
        gen_x, gen_y = next(data_gen.flow(x, y, batch_size=32))
        yield {"inputs_image": gen_x, "inputs_detection": gen_x}, gen_y  # Yield a dictionary

# Specify output signature for the generator
output_signature = (
    {"inputs_image": tf.TensorSpec(shape=(None, 480, 640, 1), dtype=tf.float32), 
     "inputs_detection": tf.TensorSpec(shape=(None, 480, 640, 1), dtype=tf.float32)},
    tf.TensorSpec(shape=(None, 480, 640), dtype=tf.float32) 
)

# Updated model.fit line to use the generator and output_signature
history = model.fit(
    tf.data.Dataset.from_generator(
        lambda: generator_two_inputs(data_gen, x_train, y_train),
        output_signature=output_signature
    ),
    steps_per_epoch=len(x_train) // 32,
    epochs=50,
    validation_data=({"inputs_image": x_val, "inputs_detection": x_val}, y_val),
    callbacks=callbacks_list
)

# Save the final model
model.save('DecideNet_mall.h5')

# Function to plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['maaae'], label='Train MAE')
    plt.plot(history.history['val_maaae'], label='Validation MAE')
    plt.title('MAE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.show()

# Plot training history after training
plot_training_history(history)

# Evaluation for table
def evaluate_model():
    reg_predictions = model.predict([x_test, np.zeros_like(x_test)])
    det_predictions = model.predict([np.zeros_like(x_test), x_test])
    combined_predictions = model.predict([x_test, x_test])

    reg_mae = K.eval(maaae(y_test, reg_predictions))
    reg_mse = K.eval(mssse(y_test, reg_predictions))
    det_mae = K.eval(maaae(y_test, det_predictions))
    det_mse = K.eval(mssse(y_test, det_predictions))
    comb_mae = K.eval(maaae(y_test, combined_predictions))
    comb_mse = K.eval(mssse(y_test, combined_predictions))

    print(f"RegNet only: MAE = {reg_mae}, MSE = {reg_mse}")
    print(f"DetNet only: MAE = {det_mae}, MSE = {det_mse}")
    print(f"RegNet+DetNet (Late Fusion): MAE = {comb_mae}, MSE = {comb_mse}")

evaluate_model()

