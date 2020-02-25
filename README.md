# Breast_Canсer_training_ScikitLearn
Logistic regression and hyperparameters customization (using Scikit-Learn) for Breast Cancer dataset.

# Description 
The aim of this notebook is to understand the process of choosing and applying the machine learning tools, comparing, selecting and improving the best models.

I analyze Breast Cancer dataset for classification in order to identify if the tumor is malignant or benign based on the cell features.
For this work I've chosen the logistic regression algorithm for classification. 

The task list which I tried to solve during the work:
1. Define hypothesis and cost functions for Logistic Regression, use fmin optimization function for finding the best parameters of the model; 
2. See how gradient descent works and how step size (learning rate) affects to it;
3. Tuning hyperparameter C with fmin function
4. Compare GridSearchCV and RandomizedSearchCV between each other for hyperparameter optimization;
5. Research how feature importance can influence on the accuracy results. For this task I apply 4 different methods - Decision Tree feature importance, information about logistic regression coefficients, RFE and SelectFromModel - and compare the results.
6. Try to find out the best fold number (k-fold) in cross validation using different values for k and LeaveOneOut cross validation method.
7. How the threshold value influence on f1 measurement and precision.
8. See how change the ROC, AUC and accuracy for different size of the control set.
9. Research how number of components in PCA influence on accuracy of the model.

For this analysis, as a guide  I use  "Introduction to Machine Learning with Python: A Guide for Data Scientists"
by Andreas C. Müller and Sarah Guido and sklearn documentation.

