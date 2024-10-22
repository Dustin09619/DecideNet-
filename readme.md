Dataset Setup Download the Mall Dataset:
The dataset should have already been extracted in /content/extracted_mall_dataset/ that contains:
frames/: Directory with images of crowd scenes in a mall.
mall_gt.mat: Ground truth density annotations.

How to Run the Code 
Step 1: Load and Prepare Data
The code consists of a data_preparation function that prepares the dataset for training by:
Loading images from the frames/ folder
Normalizing the image data
Loading the ground-truth density annotations from mall_gt.mat
Step 2: Train the Model
Run the main code DecideNet_mall.py for training of model DecideNet. It automatically loads the dataset Mall, designs the network architecture, and trains the model:
python DecideNet_mall.py
The model applies both the detection and regression branches to train an attention mechanism which is responsible for adaptively choosing the best estimation for each pixel.
The trained model will be saved as DecideNet_mall.h5.
Step 3: Visualization of Density Maps (Figure 7)
For example, image, one can run the regression-based, detection-based, and final density maps
python DecideNet_mall.py
This will print the three density maps side by side:
 
Regression-based density map
Detection-based density map
Final combined density map (attention-guided)
Step 4: Evaluate the Model (Table 4)
Now use the evaluate_model function to give you the Mean Absolute Error and Mean Squared Error for the regression branch, detection branch, and the final combined output. This should print similar results to Table 4 from the DecideNet paper.
To evaluate the model: python DecideNet_mall.py
This will compute and print out the MAE and MSE for the two branches as well as the model overall.
Step 5: Plotting Figure 6 Prediction vs. Ground Truth
Generate a plot that shows the predicted counts from different branches, compared to the ground-truth counts. Sort images according to crowd count of the ground truth.
Plot the predictions for RegNet, DetNet, and the combined model versus the ground-truth count.
This is done automatically when you run the program. python DecideNet_mall.py
Results
The following results will be obtained:
Fig Density map visualizations (regression, detection, combined).
Table Mean absolute error and mean squared error in regression-only and detection-only and the finally combined model.
Fig Predicted crowd counts against the truth.
Notes
The default configuration has been set to a constant learning rate (lr=5e-3) that trains for 50 epochs. You may change it in the build_model() function if you wish to reduce the value of the learning rates or train for more epochs to get the best result with minimal risk of overfitting.
The model architecture and loss functions are designed to mirror DecideNet closely as described in the paper, but often hyperparameter tuning beyond what is reported there will be necessary to achieve best results on individual data sets.
