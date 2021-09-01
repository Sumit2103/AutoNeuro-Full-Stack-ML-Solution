# AutoNeuro-Full-Stack-ML-Solution

Introduction

The goal here is to build an end to end automated Machine Learning solution where the user will only give the data and select the type of problem, and the result will be the best performing hyper tuned Machine Learning model. The user will also get privileges to choose the deployment options.
This project shall be delivered in two phases:
Phase 1: All the functionalities with PyPi packages.
Phase2: Integration of UI to all the functionalities. 
The technical design document gives a design blueprint of the Autoneuro project. This document communicates the technical details of the solution proposed.
In addition, this document also captures the different workflows involved to build the solution, exceptions in the workflows and any assumptions that have been considered. 
Once agreed as the basis for the building of the project, the flowchart and assumptions will be used as a platform from which the solution will be designed.
Changes to this business process may constitute a request for change and will be subject to the agreed agility program change procedures.
Note: All the code will be written in python version 3.7

High level objectives

The high-level objectives are:
1.	Enable reading/loading of data from the various sources and convert them into pandas dataframe(details mentioned in the Data Ingestion Section).
2.	Enable reading various file formats and convert them into pandas dataframe(details mentioned in the Data Ingestion Section).
3.	Give user the option to specify feature and target columns.
4.	Give user the option to select the problem type, viz. Regression, Classification (include anomaly detection), Clustering or Time Series. 
5.	Perform statistical analytics of the data and prepare a table for the analysis and show it on screen.
6.	Perform graphical analysis for the data and Showcase the results (graphs) on the screen.
7.	Perform data cleaning operation with all the steps required and showcase a report on screen.
8.	 After data cleaning showcase the graphical analysis once again for comparison.
9.	Check whether clustering is required or not.
10.	Choose the appropriate ML model for training.
11.	Perform model Tuning.
12.	Create a list of top 3 models  and show multiple metrics for them.
13.	Give option for prediction.
14.	Give options for docker container creation.
15.	Give option for automatic cloud deployment.

Data Profiling
After reading the data, automatically the following details should be shown:
a)	The number of rows
b)	The number of columns
c)	Number of missing values per column and their percentage
d)	Total missing values and it’s percentage
e)	Number of categorical columns and their list
f)	Number of numerical columns and their list
g)	Number of duplicate rows
h)	Number of columns with zero standard deviation and their list
i)	Size occupied in RAM

Stats Based EDA

MVP
OLS
VIF
Correlation
Phase1:
Column contributions/ importance
Annova Test
Chi Square test
Z test
T -test
Weight of Evidence 
F – Test
Phase 2:
Seasonality
Stationary Data

Graph-Based EDA

Create the following graphs:
MVP:
Correlation Heatmaps
Check for balance/imbalance
Phase1:
Count plots
Boxplot for outliers
Piecharts for categories
Geographical plots for scenarios
Line charts for  trends
Barplots
Area Charts
KDE Plots
Stacked charts
Scatterplot
Phase 2:
Word maps
PACF
ACF
Add Custom controls sliders etc

Note: We are going to use plotly for all the graphs.( https://plotly.com/python/)

Data Transformers( Pre-processing steps)

MVP:
Null value handling
Categorical to numerical
Imbalanced data set handling
Handling columns with std deviation zero or below a threshold
Normalisation
PCA
Phase1:
Outlier detection
Data Scaling/ Normalisation
Feature Selection: https://scikit-learn.org/stable/auto_examples/index.html#feature-selection

ML Model Selection

MVP:
3 Models—KNN, RandomForest, XGBoost
Phase1:
Model Selection criteria

Model Tuning and Optimization

Note: The data should have been divided into train and validation set before this.
Methods for hyper tuning all kinds of models.
Regression:
Linear Regression
Decision Tree
Random Forest
XG Boost
Support Vector Regressor
KNN Regressor

Model selection criteria:

MSE, RMSE, R squared, adjusted R squared
Classification:
Logistic Regression
Decision Tree
Random Forest
XG Boost
Support Vector Classifier
KNN Classifier
Naïve Baye’s

Model selection criteria:
Accuracy, AUC, Precision, Recall, F Beta


Clustering:

K-Means
Hierarchial
DBSCAN
Phase 2:
GLM
GAM (https://www.statsmodels.org/stable/regression.html)
Time Series
Anomaly Detection
Novelty Detection
Optics
Gaussian Mixtures
BIRCH
NLP
Deep Learning
Regularization modules if necessary

Testing Modules

Divide the training data itself into  train and test sets
Use test data to have tests run on the three best models
Give the test report
a)	R2 Score
b)	Adjusted R2 score
c)	MSE
d)	Accuracy
e)	Precision
f)	Recall
g)	F Beta
h)	Cluster Purity
i)	Silhouette score 

Phase 2
AIC
BIC

Prediction Pipeline  

Use the existing data read modules
Use the existing pre-processing module
Load the model into memory
Do predictions
Store  prediction results(show sample predictions)
Phase 2:
UI for predictions

Deployment Strategy 

Take the cloud name as input
Prepare the metadata files based on cloud
Phase 2:
Accept the user credentials
Prepare a script file to push changes
Docker instance
Push of the docker instance to cloud


Monitoring

Phase 2
No. Of predictions for individual classes
No. of  predictions (per day, per hour, per week etc.)
No. of hits
Training data size (number of rows)
Time spent in training
Failures

15	Options for Logging in DB Logging
Separate Folder for logs
Logging of every step
Entry to the methods
Exit from the methods with success/ failure message
Error message Logging
Model comparisons
Training start and end
Prediction start and end
Achieve asynchronous logging

Phase 2:

Options for Log Publish

Requirements for model training

The minimum configuration should be:
•	8 GB RAM
•	2 GB of Hard Disk Space
•	Intel Core i5 Processor

Requirements for model testing

The minimum configuration should be:
•	4 GB RAM
•	2 GB of Hard Disk Space
•	Intel Core i5 Processor




