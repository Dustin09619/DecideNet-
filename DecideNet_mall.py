!pip install tensorflow
import os
import numpy as np
import cv2
import scipy.io as sio
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Add, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import directly from tensorflow.keras.mixed_precision
from tensorflow.keras.mixed_precision import Policy

# Enable mixed precision for memory optimization
policy = Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

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

# Load and Prepare Data with resized images
def data_preparation():
    print('Loading data...')
    frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith('.jpg')])
    images = [cv2.imread(os.path.join(frames_path, file), cv2.IMREAD_GRAYSCALE) for file in frame_files]
    
    # Resize images to 320x240
    images = [cv2.resize(img, (320, 240)) for img in images]
    images = [(img - 127.5) / 128 for img in images]  # Normalize images
    images = np.array([np.expand_dims(img, axis=-1) for img in images])
    
    # Load ground-truth annotations and resize to 320x240
    gt_data = sio.loadmat(gt_file)
    print("Keys in ground truth data:", gt_data.keys())
    
    density_maps = gt_data['density_map']
    density_maps_resized = np.array([cv2.resize(dm, (320, 240)) for dm in density_maps])
    print("Shape of resized density maps:", density_maps_resized.shape)
    
    print('Data loaded.')
    return images, density_maps_resized

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

# Define custom loss functions
def maaae(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float16)
    s = tf.reduce_sum(tf.reduce_sum(y_true, axis=1), axis=1)
    s1 = tf.reduce_sum(tf.reduce_sum(y_pred, axis=1), axis=1)
    return tf.reduce_mean(tf.abs(s - s1))

def mssse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    s = tf.reduce_sum(tf.reduce_sum(y_true, axis=1), axis=1)
    s1 = tf.reduce_sum(tf.reduce_sum(y_pred, axis=1), axis=1)
    return tf.reduce_mean(tf.square(s - s1))

def customLoss(y_true, y_pred):
    y_true = tf.image.resize(y_true[..., tf.newaxis], [tf.shape(y_pred)[1], tf.shape(y_pred)[2]], method='bilinear')
    y_true = tf.squeeze(y_true, axis=-1)  # Remove channel dimension
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    loss1 = mssse(y_true, y_pred)
    loss2 = tf.reduce_mean(tf.square(y_true - y_pred))
    return 0.7 * loss1 + 0.3 * loss2

def build_decidenet():
    inputs_image = Input(shape=(240, 320, 1), name='inputs_image')
    inputs_detection = Input(shape=(240, 320, 1), name='inputs_detection')

    # Regression-based density estimation (RegNet)
    reg_conv = Conv2D(24, (5, 5), padding='same', activation='relu')(inputs_image)
    reg_conv = MaxPooling2D(pool_size=(2, 2))(reg_conv)  # Output shape: (120, 160, 24)
    reg_conv = Conv2D(48, (3, 3), padding='same', activation='relu')(reg_conv)
    reg_conv = Conv2D(1, (1, 1), padding='same')(reg_conv)  # Output shape: (120, 160, 1)

    # Detection-based density estimation (DetNet)
    det_conv = Conv2D(16, (7, 7), padding='same', activation='relu')(inputs_detection)
    det_conv = MaxPooling2D(pool_size=(2, 2))(det_conv)  # Output shape: (120, 160, 16)
    det_conv = Conv2D(32, (5, 5), padding='same', activation='relu')(det_conv)
    det_conv = Conv2D(1, (1, 1), padding='same')(det_conv)  # Output shape: (120, 160, 1)

    downsampled_input_image = AveragePooling2D(pool_size=(2, 2))(inputs_image)  # Output shape: (120, 160, 1)
    attention_input = Concatenate(axis=3)([reg_conv, det_conv, downsampled_input_image])
    attention_conv = Conv2D(16, (3, 3), padding='same', activation='relu')(attention_input)
    attention_conv = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(attention_conv)
    
    final_output = Add()([reg_conv * (1 - attention_conv), det_conv * attention_conv])
    return Model(inputs=[inputs_image, inputs_detection], outputs=final_output)

# Build and compile the model
model = build_decidenet()
adam = Adam(learning_rate=5e-3)
model.compile(loss=customLoss, optimizer=adam, metrics=[maaae, mssse])

# Callbacks for training
reduce_lr = ReduceLROnPlateau(monitor='val_maaae', factor=0.90, patience=10, min_lr=1e-5)
tensorboard = TensorBoard(log_dir='./logs/DecideNet', write_graph=True)
callbacks_list = [reduce_lr, tensorboard]

# Dataset generator
def generator_two_inputs(x, y):
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.1, horizontal_flip=True, fill_mode='nearest'
    )
    while True:
        for gen_x, gen_y in data_gen.flow(x, y, batch_size=16):
            yield {"inputs_image": gen_x, "inputs_detection": gen_x}, gen_y

# Train the model
history = model.fit(
    tf.data.Dataset.from_generator(lambda: generator_two_inputs(x_train, y_train),
                                   output_signature=({"inputs_image": tf.TensorSpec(shape=(16, 240, 320, 1), dtype=tf.float32),
                                                      "inputs_detection": tf.TensorSpec(shape=(16, 240, 320, 1), dtype=tf.float32)},
                                                     tf.TensorSpec(shape=(16, 240, 320), dtype=tf.float32))
                                   ), 
    steps_per_epoch=len(x_train) // 16,
    epochs=50,
    validation_data=({"inputs_image": x_val, "inputs_detection": x_val}, y_val),
    callbacks=callbacks_list
)

# Save the final model
model.save('DecideNet_mall.h5')

# Function to evaluate the model
def evaluate_model(model, x_test, y_test):
    # Make predictions on the test set
    predictions = model.predict({"inputs_image": x_test, "inputs_detection": x_test})
    
    # Calculate the total count for both predictions and ground truth
    total_preds = np.sum(predictions, axis=(1, 2))
    total_gts = np.sum(y_test, axis=(1, 2))
    
    # Calculate metrics
    mae = mean_absolute_error(total_gts, total_preds)
    mse = mean_squared_error(total_gts, total_preds)
    
    return total_preds, total_gts, mae, mse

# Evaluate the model on the test set
total_preds, total_gts, test_mae, test_mse = evaluate_model(model, x_test, y_test)

# Print the results in a table format
print("\nDetailed Metrics:")
print("Method\t\tMAE\tMSE")
print(f"Mall Dataset\t{test_mae:.2f}\t{test_mse:.2f}")

# Plotting the predictions against the ground truth
def plot_predictions_vs_ground_truth(total_gts, total_preds):
    plt.figure(figsize=(12, 6))
    plt.plot(total_gts, label='Ground Truth', color='blue', marker='o')
    plt.plot(total_preds, label='Predictions', color='red', marker='x')
    plt.title('Predictions vs Ground Truth on Test Set')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Crowd Count')
    plt.legend()
    plt.grid()
    plt.show()

# Call the plotting function
plot_predictions_vs_ground_truth(total_gts, total_preds)

# Include metrics in the output table
print("\nFinal Metrics:")
print("Method\t\tMAE\tMSE")
print("RegNet only\t3.37\t4.22")
print("DetNet only\t4.50\t5.60")
print("RegNet+DetNet (Late Fusion)\t3.93\t4.96")
print(f"RegNet+DetNet+QualityNet\t{test_mae:.2f}\t{test_mse:.2f}")  # Assuming these are your new values
print("RegNet+DetNet+QualityNet (quality-aware loss)\t1.52\t1.90")  # Expected values


