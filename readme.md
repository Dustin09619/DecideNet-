DecideNet for Mall Crowd Counting Dataset Setup Download the Mall Dataset : The dataset should already have been extracted in /content/extracted_mall_dataset/ that contains frames/: Directory with images of crowd scenes in a mall. mall_gt.mat : Ground truth density annotations. To Run the Code Step 1: Load and prepare data The code contains the data_preparation function for preparing the dataset to the training process by: End.
Loading the ground-truth density annotations from mall_gt.mat.
Step 2: Training
Running the main code DecideNet_mall.py trains the model DecideNet. It loads the dataset Mall, designs the network architecture, and then automatically trains it:
The model uses a detection as well as a regression head to train up an attention mechanism which makes adaptively the best selection for a given pixel in an image. It will save as DecideNet_mall.h5.

Step 3: Visualization of Density Maps Figure 7

For instance, one can call the regressions-based density map by running the commands as
```
 DecideNet_mall.py
It will print side-by-side the three the density map
Regression-based density map
Detection-based density map
Final attention-guided combined density map
Model Evaluation (Table 4)
Now apply the function evaluate_model() to compute and print you the MAE and MSE for each the regression, detection and combined output.
This would then print a similar answer as in Table 4 found on the paper of DecideNet.
 

To make evaluation on model:
  
 DecideNet_mall.py
This will compute and print the MAE and MSE of the two branches as well as the overall model.

Step 5: Figure 6 Prediction vs. Ground Truth Plot
Produce a plot of predicted counts from all the different branches against ground truth counts. Sort images by ground truth crowd count.

This is done automatically when you run the program:

 DecideNet_mall.py
Results
The results are as follows:

Figure: Density map visualizations (regression, detection, combined).
Table: Mean absolute error and mean squared error in regression-only and detection-only models, and the finally combined model.
Figure: Predicted crowd counts against the ground truth.
Notes
By default, the learning rate configuration has been fixed to constant for 50 epochs at a value of lr=5e-3. To decrease the learning rates values or train longer to acquire the best results with lower risk of overfitting, you could change the build_model() function.
Model architectures and loss functions are intended to be as close as possible to DecideNet while described in the paper but often hyperparameter tuning beyond reported there will be required to produce the best results on single datasets.
Additional Notes
Replace dummy detection maps by real detection results for an honest evaluation.
Adjust your paths according to your specific dataset structure if necessary
Tune hyperparameters for epochs and learning rate appropriate to maximize performance.
