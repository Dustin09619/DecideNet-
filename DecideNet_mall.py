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

# Enable mixed precision for memory optimization
from tensorflow.keras.mixed_precision import Policy
policy = Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Define paths
dataset_path = '/content/drive/MyDrive/mall_dataset'
frames_path = os.path.join(dataset_path, 'frames')
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
    
    print('Data loaded.')
    return images, np.array([cv2.resize(dm, (320, 240)) for dm in density_maps])

import os
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

for i, density_map in enumerate(y_data[:10]):  # Check the first 10 samples
       print(f"Image {i} - Ground truth crowd count: {np.sum(density_map)}")

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, Lambda
from tensorflow.keras.models import Model

def maaae(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    s = tf.reduce_sum(tf.reduce_sum(y_true, axis=1), axis=1)
    s1 = tf.reduce_sum(tf.reduce_sum(y_pred, axis=1), axis=1)
    return tf.reduce_mean(tf.abs(s - s1))

def mssse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    s = tf.reduce_sum(tf.reduce_sum(y_true, axis=1), axis=1)
    s1 = tf.reduce_sum(tf.reduce_sum(y_pred, axis=1), axis=1)
    return tf.reduce_mean(tf.square(s - s1))
    
def custom_loss(y_true, y_pred):
    # Ensure the shape of y_true matches y_pred
    y_true = tf.ensure_shape(y_true, [None, None, None, 1])  
    y_true = tf.image.resize(y_true, [tf.shape(y_pred)[1], tf.shape(y_pred)[2]], method='bilinear')

    # Calculate the density loss and quality loss
    density_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    quality_loss = mssse(y_true, y_pred)  # Consider adjusting how quality loss is computed
    # Introduce a weighting factor
    return density_loss + 0.5 * quality_loss  # Adjust the weight for quality loss

# Adjust the model architecture if needed
def build_decidenet():
    inputs_image = Input(shape=(240, 320, 1), name='inputs_image')
    inputs_detection = Input(shape=(240, 320, 1), name='inputs_detection')

    # RegNet
    reg_conv = Conv2D(32, (5, 5), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs_image)
    reg_conv = MaxPooling2D(pool_size=(2, 2))(reg_conv)
    reg_conv = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(reg_conv)
    reg_conv = Conv2D(1, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(reg_conv)

    # DetNet
    det_conv = Conv2D(24, (7, 7), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs_detection)
    det_conv = MaxPooling2D(pool_size=(2, 2))(det_conv)
    det_conv = Conv2D(48, (5, 5), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(det_conv)
    det_conv = Conv2D(1, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(det_conv)

    # Attention Mechanism (QualityNet)
    downsampled_input_image = AveragePooling2D(pool_size=(2, 2))(inputs_image)
    attention_input = Concatenate(axis=3)([reg_conv, det_conv, downsampled_input_image])
    attention_conv = Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005))(attention_input)
    attention_conv = Dropout(0.5)(attention_conv)
    attention_map = tf.keras.activations.sigmoid(attention_conv)

    # Weighted sum of RegNet and DetNet predictions
    alpha = tf.Variable(initial_value=0.5, trainable=True, dtype=tf.float32)  # Introduce learnable weight
    final_output = Conv2D(1, (1, 1), padding='same')(
        Add()([reg_conv * (1 - attention_map), det_conv * attention_map])
    ) 

    return Model(inputs=[inputs_image, inputs_detection], outputs=Lambda(lambda x: x * 1000)(final_output))
   
   
  

# Custom quality-aware loss function

def custom_loss(y_true, y_pred, initial_weight=0.5, decay_rate=0.95):
    y_pred = tf.image.resize(y_pred, [tf.shape(y_true)[1], tf.shape(y_true)[2]], method='bilinear')  
    y_true = y_true * 1000
    y_pred = y_pred * 1000
    density_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    quality_loss = mssse(y_true, y_pred)
    weight = initial_weight * (decay_rate ** tf.cast(tf.keras.backend.get_value(model.optimizer.iterations), tf.float32))
    return density_loss + weight * quality_loss

# Compile the DecideNet model
model = build_decidenet()
adam = Adam(learning_rate=5e-4)  # Learning rate
model.compile(loss=custom_loss, optimizer=adam, metrics=[maaae, mssse]) 

# Callbacks for training - Adjust patience and reduce_lr parameters
callbacks_list = [
    ReduceLROnPlateau(monitor='val_maaae', factor=0.5, patience=3, min_lr=1e-6),  # Adjust patience and factor
    EarlyStopping(monitor='val_maaae', mode='min', patience=40, restore_best_weights=True),
    TensorBoard(log_dir='./logs/DecideNet_optimized', write_graph=True),
    ModelCheckpoint('DecideNet_best_model.keras', monitor='val_maaae', save_best_only=True, mode='min', verbose=1)
]
def generator_two_inputs(x, y):
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        zoom_range=0.2, horizontal_flip=True, fill_mode='nearest',
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
            
history = model.fit(
    tf.data.Dataset.from_generator(
        lambda: generator_two_inputs(x_train, y_train),
        output_signature=(
            {"inputs_image": tf.TensorSpec(shape=(16, 240, 320, 1), dtype=tf.float32),
             "inputs_detection": tf.TensorSpec(shape=(16, 240, 320, 1), dtype=tf.float32)},
            tf.TensorSpec(shape=(16, 240, 320, 1), dtype=tf.float32)
        )
    ),
    steps_per_epoch=len(x_train) // 16,
    epochs=150,
    validation_data=({"inputs_image": x_val, "inputs_detection": x_val}, np.expand_dims(y_val, axis=-1)),
    callbacks=callbacks_list
)

# Save the final model
model.save('DecideNet_mall_optimized.h5')

model = build_decidenet()
model.summary()
for layer in model.layers:
    print(layer.name)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Model

# Define a function to compute MAE and MSE
def evaluate_metrics(y_true, y_pred):
    if y_pred.shape != y_true.shape:
        y_pred = tf.image.resize(y_pred, [y_true.shape[1], y_true.shape[2]], method='bilinear').numpy()
        y_pred = y_pred.squeeze()  # Remove the channel dimension if it exists

    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    return mae, mse

# Define a function to generate predictions in batches to avoid memory overload
def batch_predict(model, x_data, batch_size=32):
    predictions = []
    for i in range(0, len(x_data), batch_size):
        batch_x = x_data[i:i + batch_size]
        batch_pred = model.predict({"inputs_image": batch_x, "inputs_detection": batch_x})
        predictions.append(batch_pred)
    return np.vstack(predictions)

# Adjust layer indexing based on current model layers for RegNet and DetNet outputs
regnet_output_layer = model.get_layer('conv2d_10').output  # RegNet output
detnet_output_layer = model.get_layer('conv2d_13').output  # DetNet output

# Create separate models for RegNet and DetNet outputs
regnet_only_model = Model(inputs=model.input, outputs=regnet_output_layer)
detnet_only_model = Model(inputs=model.input, outputs=detnet_output_layer)

# Generate predictions for RegNet, DetNet, and full DecideNet in smaller batches
batch_size = 32
regnet_predictions = batch_predict(regnet_only_model, x_test, batch_size=batch_size)
detnet_predictions = batch_predict(detnet_only_model, x_test, batch_size=batch_size)
decidenet_predictions = batch_predict(model, x_test, batch_size=batch_size)

# Calculate metrics for each component
regnet_mae, regnet_mse = evaluate_metrics(y_test, regnet_predictions)
detnet_mae, detnet_mse = evaluate_metrics(y_test, detnet_predictions)
late_fusion_predictions = 0.5 * (regnet_predictions + detnet_predictions)
late_fusion_mae, late_fusion_mse = evaluate_metrics(y_test, late_fusion_predictions)
decidenet_mae, decidenet_mse = evaluate_metrics(y_test, decidenet_predictions)

# Display Table 4 results
table4_results = pd.DataFrame({
    "Model Component": ["RegNet Only", "DetNet Only", "RegNet + DetNet (Late Fusion)", "RegNet + DetNet + QualityNet"],
    "MAE": [regnet_mae, detnet_mae, late_fusion_mae, decidenet_mae],
    "MSE": [regnet_mse, detnet_mse, late_fusion_mse, decidenet_mse]
})
print("Table 4: Qualitative results of DecideNet components on the Mall dataset")
print(table4_results)

# Figure 6: Prediction vs. Ground-Truth Plot for Test Samples
true_counts = [np.sum(y) for y in y_test]
predicted_counts = [np.sum(y) for y in decidenet_predictions]

# Sort by true counts for a smoother plot
sorted_indices = np.argsort(true_counts)
true_counts_sorted = np.array(true_counts)[sorted_indices]
predicted_counts_sorted = np.array(predicted_counts)[sorted_indices]

# Plot Figure 6
plt.figure(figsize=(10, 5))
plt.plot(true_counts_sorted, label="Ground Truth", color="red", linestyle='-', linewidth=1.5)
plt.plot(predicted_counts_sorted, label="Prediction", color="cyan", linestyle='--', linewidth=1.5)
plt.xlabel("Image ID (sorted by ground-truth count)")
plt.ylabel("Crowd Count")
plt.title("Figure 6: Prediction and Ground-Truth Crowd Counts on Test Set (Mall)")
plt.legend()
plt.show()

# Figure 7: Visualization of Density Maps from RegNet, DetNet, and DecideNet
sample_idx = 10  # Choose any index within the test set range
sample_image = x_test[sample_idx:sample_idx + 1]

# Generate density map predictions
regnet_density_map = regnet_only_model.predict({"inputs_image": sample_image, "inputs_detection": sample_image})
detnet_density_map = detnet_only_model.predict({"inputs_image": sample_image, "inputs_detection": sample_image})
decidenet_density_map = model.predict({"inputs_image": sample_image, "inputs_detection": sample_image})

# Plot Figure 7
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(sample_image[0, :, :, 0], cmap='gray')
plt.title("Input Image")

plt.subplot(1, 4, 2)
plt.imshow(regnet_density_map[0, :, :, 0], cmap='jet')
plt.title("RegNet Density Map")
plt.colorbar()

plt.subplot(1, 4, 3)
plt.imshow(detnet_density_map[0, :, :, 0], cmap='jet')
plt.title("DetNet Density Map")
plt.colorbar()

plt.subplot(1, 4, 4)
plt.imshow(decidenet_density_map[0, :, :, 0], cmap='jet')
plt.title("Final DecideNet Density Map")
plt.colorbar()

plt.suptitle("Figure 7: Visualization of Density Maps (Mall Dataset)")
plt.show()

    


