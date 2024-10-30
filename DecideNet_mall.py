import os
import numpy as np
import cv2
import scipy.io as sio
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Add, Activation
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    # Load images
    frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith('.jpg')])
    images = [cv2.imread(os.path.join(frames_path, file), cv2.IMREAD_GRAYSCALE) for file in frame_files]
    images = [(img - 127.5) / 128 for img in images]  # Normalize images
    images = np.array([np.expand_dims(img, axis=-1) for img in images])
    
    # Load ground-truth annotations
    gt_data = sio.loadmat(gt_file)
    print("Keys in ground truth data:", gt_data.keys())

    density_maps = gt_data['density_map']  # Assuming density maps are stored in 'density_map'
    
    print('Data loaded.')
    return images, density_maps

# Load data
x_data, y_data = data_preparation()

# Split data (800 training, 100 validation from training, 1200 testing)
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
def maaae(y_true, y_pred):
    s = K.sum(K.sum(y_true, axis=1), axis=1)
    s1 = K.sum(K.sum(y_pred, axis=1), axis=1)
    return K.mean(K.abs(s - s1))

def mssse(y_true, y_pred):
    s = K.sum(K.sum(y_true, axis=1), axis=1)
    s1 = K.sum(K.sum(y_pred, axis=1), axis=1)
    return K.mean(K.square(s - s1))

def customLoss(y_true, y_pred):
    loss1 = mssse(y_true, y_pred)
    loss2 = K.mean(K.square(y_true - y_pred))
    return 0.7 * loss1 + 0.3 * loss2

# Define the DecideNet architecture with enhanced QualityNet
def build_decidenet():
    # Inputs
    inputs_image = Input(shape=(None, None, 1))
    inputs_detection = Input(shape=(None, None, 1))

    # Regression-based density estimation (RegNet)
    reg_conv = Conv2D(24, (5, 5), padding='same', activation='relu')(inputs_image)
    reg_conv = MaxPooling2D(pool_size=(2, 2))(reg_conv)
    reg_conv = Conv2D(48, (3, 3), padding='same', activation='relu')(reg_conv)
    reg_conv = Conv2D(1, (1, 1), padding='same')(reg_conv)

    # Detection-based density estimation (DetNet)
    det_conv = Conv2D(16, (7, 7), padding='same', activation='relu')(inputs_detection)
    det_conv = MaxPooling2D(pool_size=(2, 2))(det_conv)
    det_conv = Conv2D(32, (5, 5), padding='same', activation='relu')(det_conv)
    det_conv = Conv2D(1, (1, 1), padding='same')(det_conv)

    # Enhanced Attention Mechanism (QualityNet)
    attention_input = Concatenate(axis=3)([reg_conv, det_conv, inputs_image])
    attention_conv = Conv2D(32, (3, 3), padding='same', activation='relu')(attention_input)
    attention_conv = Conv2D(16, (3, 3), padding='same', activation='relu')(attention_conv)
    attention_conv = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(attention_conv)

    # Combine results based on attention
    final_output = Add()([reg_conv * (1 - attention_conv), det_conv * attention_conv])
    
    # Create the model
    model = Model(inputs=[inputs_image, inputs_detection], outputs=final_output)
    return model

# Build and compile the model
model = build_decidenet()
model.summary()
adam = Adam(learning_rate=5e-3)
model.compile(loss=customLoss, optimizer=adam, metrics=[maaae, mssse])

# Training setup
reduce_lr = ReduceLROnPlateau(monitor='val_maaae', factor=0.90, patience=10, min_lr=1e-5)
tensorboard = TensorBoard(log_dir='./logs/DecideNet', write_graph=True)
callbacks_list = [reduce_lr, tensorboard]

# Train the model with data augmentation
history = model.fit(
    data_gen.flow(x_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=callbacks_list
)

# Save the final model
model.save('DecideNet_mall.h5')

# Visualization for Density Maps
def visualize_density_maps(image, regression_map, detection_map, combined_map):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title('Regression-based Density')
    plt.imshow(regression_map, cmap='hot')
    plt.subplot(1, 3, 2)
    plt.title('Detection-based Density')
    plt.imshow(detection_map, cmap='hot')
    plt.subplot(1, 3, 3)
    plt.title('Combined Density (Attention)')
    plt.imshow(combined_map, cmap='hot')
    plt.show()

# Example visualization
example_idx = 0
example_image = x_val[example_idx]
regression_map = model.predict([example_image[np.newaxis, ...], np.zeros_like(example_image)[np.newaxis, ...]])[0, ..., 0]
detection_map = model.predict([np.zeros_like(example_image)[np.newaxis, ...], example_image[np.newaxis, ...]])[0, ..., 0]
combined_map = model.predict([example_image[np.newaxis, ...], example_image[np.newaxis, ...]])[0, ..., 0]
visualize_density_maps(example_image[..., 0], regression_map, detection_map, combined_map)

# Evaluation for table
def evaluate_model():
    reg_predictions = model.predict([x_test, np.zeros_like(x_test)])
    det_predictions = model.predict([np.zeros_like(x_test), x_test])
    combined_predictions = model.predict([x_test, x_test])

    reg_mae = K.eval(maaae(y_test, reg_predictions))
    reg_mse = K.eval(mssse(y_test, reg_predictions))
    det_mae = K.eval(maaae(y_test, det_predictions))
    det_mse = K.eval(mssse(y_test, det_predictions))
    final_mae = K.eval(maaae(y_test, combined_predictions))
    final_mse = K.eval(mssse(y_test, combined_predictions))

    print(f"Regression-only MAE: {reg_mae:.4f}, MSE: {reg_mse:.4f}")
    print(f"Detection-only MAE: {det_mae:.4f}, MSE: {det_mse:.4f}")
    print(f"Combined Model MAE: {final_mae:.4f}, MSE: {final_mse:.4f}")

evaluate_model()

# Plot Predictions vs. Ground Truth
def plot_predictions_vs_ground_truth():
    predicted_counts_regression = [np.sum(m) for m in reg_predictions]
    predicted_counts_detection = [np.sum(m) for m in det_predictions]
    predicted_counts_combined = [np.sum(m) for m in combined_predictions]
    ground_truth_counts = [np.sum(y) for y in y_test]

    plt.figure(figsize=(10, 5))
    plt.plot(ground_truth_counts, label="Ground Truth Counts", color="black")
    plt.plot(predicted_counts_regression, label="Regression-only", linestyle='--', color="blue")
    plt.plot(predicted_counts_detection, label="Detection-only", linestyle=':', color="green")
    plt.plot(predicted_counts_combined, label="Combined Model", linestyle='-', color="red")
    plt.xlabel("Test Samples")
    plt.ylabel("Crowd Count")
    plt.legend()
    plt.title("Predicted vs. Ground Truth Counts")
    plt.show()

plot_predictions_vs_ground_truth()
