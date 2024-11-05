DecideNet for Mall Crowd Counting Dataset

Dataset Setup
Download the Mall Dataset and ensure it is extracted in the following format:

/content/extracted_mall_dataset/
frames/: Directory of images for crowd scenes in a mall.
mall_gt.mat: Contains the ground truth density annotations.

Steps to Run the Code
Step 1: Load and Prepare Data
The data_preparation() function in the code prepares the dataset for training by:

Loading the ground-truth density annotations from mall_gt.mat
Resizing and normalizing the images.

Step 2: Training the Model
Run the top program, EnhancedDecidenet.py, which:

Loads the Mall dataset
Designs and initializes the DecideNet architecture
Trains the network using two heads simultaneously:
One head for detection
Another for regression, incorporating the attention module
After training, DecideNet_mall.h5 will be saved.

Density Map Visualization
Once training completes, density maps for each branch in the network pipeline should be visible.
Run the following commands in the terminal to visualize:
python EnhancedDecidenet.py
This will output both the regression-based and detection-based density maps, visualized together with the combined, attention-driven density map.

Step 3: Model Evaluation (Table 4)
Compute and print Mean Absolute Error (MAE) and Mean Squared Error (MSE) for all three output types using the evaluate_model() function:

Regression Output
Detection Output
Combined Output
Run the following in the terminal to evaluate and print MAE and MSE for each branch and the overall model:
python EnhancedDecidenet.py

Step 4: Predicted vs. Ground Truth Plot (Figure 6)
This step generates a plot that shows, for each branch, the predicted crowd counts alongside the ground truth counts, ordered by value.
Generate this plot by running:

python EnhancedDecidenet.py

Output
The final results include:

Figure: Density map visualization from regression, detection, and the corresponding attention-guided combined map.
Table: MAE and MSE values for:
Regression-only model
Detection-only model
Model with combined attention mechanism
Figure: Predicted crowd counts compared to ground truth counts.

Comments
Learning Rate: Use a constant learning rate of and epochs accordingly.
If this rate doesn’t work for your application, adjust it in the build_model() function. Lower learning rates and more epochs can reduce overfitting, leading to better results.
Model Architecture and Loss Functions: The architecture and loss function are similar to DecideNet. Fine-tuning hyperparameters may be necessary to achieve optimal performance on specific datasets.
More Tips
Detection Map: Use actual detection maps instead of placeholders for accurate evaluation.
Dataset Paths: Modify dataset paths in the code to match your system setup.
Hyperparameter Tuning: Experiment with hyperparameters like epochs and learning rate to enhance model performance and reduce overfitting.
