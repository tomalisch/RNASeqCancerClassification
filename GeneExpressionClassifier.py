## Using public labeled gene expression data to predict cancer type using different models
# 1) Load and clean data
# 2) Balance groups and create data loaders
# 3) Build, train, validate, and test accuracy across different ML models
# 4) Print output and conclude

## Dependencies
import pydna as PyDNA
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE


# Load in data .csv as dataframes
path = os.getcwd()
filepathX = os.path.join( path, "TCGA-PANCAN-HiSeq-801x20531/", "data.csv")
filepathY = os.path.join( path, "TCGA-PANCAN-HiSeq-801x20531/", "labels.csv")

xData = pd.read_csv(filepathX)
yData = pd.read_csv(filepathY)

# Drop NaNs and redundant columns from dataframes
xData.dropna(inplace=True)
xData.drop(columns=["Unnamed: 0"], inplace=True)
yData.dropna(inplace=True)
yData.drop(columns=["Unnamed: 0"], inplace=True)

# Split data into train and test sets (80/20 split)
X_train, X_test, Y_train, Y_test = train_test_split( xData, yData, test_size=0.3 )

## Create function for cross validation pipeline
def grid_pipe(clf, X, Y, params):
    pipe = Pipeline([ ('sampling', SMOTE()), ('stdsc', stdsc), ('classifier', clf) ])
    score = { 'AUC':'roc_auc', 
           'RECALL':'recall',
           'PRECISION':'precision',
           'F1':'f1' }
    
    gCV = GridSearchCV( estimator=pipe, param_grid=params, cv=5, scoring=score, n_jobs=12, refit='F1',
                       return_train_score=True)
    gCV.fit( X, Y )

    return gCV

# Create list of classifiers to test
classifiers = [('Logistic Regression',
  LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                     intercept_scaling=1, l1_ratio=None, max_iter=100,
                     multi_class='auto', n_jobs=None, penalty='l2',
                     random_state=None, solver='newton-cholesky', tol=0.0001, verbose=0,
                     warm_start=False)),
 ('RandomForest',
  RandomForestClassifier()),

 ('Gradient Boosting Classifier',
  GradientBoostingClassifier(criterion='friedman_mse', init=None,
                             learning_rate=0.1, loss='deviance', max_depth=3,
                             max_features=None, max_leaf_nodes=None,
                             min_impurity_decrease=0.0,
                             min_samples_leaf=1, min_samples_split=2,
                             min_weight_fraction_leaf=0.0, n_estimators=100,
                             n_iter_no_change=None,
                             random_state=None, subsample=1.0, tol=0.0001,
                             validation_fraction=0.1, verbose=0,
                             warm_start=False))]

# Create list of tunable parameters'
params = [{'classifier__penalty': ('l1', 'l2'), 'classifier__C': (0.01, 0.1, 1.0, 10)},
 {'classifier__n_neighbors': (10, 15, 25)},
 {'classifier__n_estimators': (80, 100, 150, 200), 'min_samples_split': (5, 7, 10, 20)}]

# Loop through classifiers and parameters in grid search
for param, classifier in zip(params, classifiers):
    print("Working on {}...".format(classifier[0]))
    clf = grid_pipe(classifier[1], X_train, Y_train, param) 
    print("Best parameter for {} is {}".format(classifier[0], clf.best_params_))
    print("Best `F1` for {} is {}".format(classifier[0], clf.best_score_))
    print('-'*50)
    print('\n')