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

from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF, PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score


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

# Initialize classifier object for parameter optimization grid search
gb = GradientBoostingClassifier(criterion='friedman_mse', init=None,
                             learning_rate=0.1, loss='deviance', max_depth=3,
                             max_features=None, max_leaf_nodes=None,
                             min_impurity_decrease=0.0,
                             min_samples_leaf=1, min_samples_split=2,
                             min_weight_fraction_leaf=0.0, n_estimators=100,
                             n_iter_no_change=None,
                             random_state=None, subsample=1.0, tol=0.0001,
                             validation_fraction=0.1, verbose=0,
                             warm_start=False)

# Create list of tunable parameters for grid search
params = [
 {'classifier__max_depth': (1, 3, 6)},
 {'classifier__min_impurity_decrease': (0.0, 0.01, 0.1)},{'classifier__min_samples_leaf': (1, 2, 3)}, {'classifier__min_samples_split': (2, 5, 10)}
 , {'classifier__min_weight_fraction_leaf': (0.0, 0.01, 0.1)}, {'classifier__n_estimators': (10, 100, 200)}, {'classifier__tol': (0.00001, 0.0001, 0.001)} ]

pipe = Pipeline([ ('sampling', SMOTE()), ('stdsc', StandardScaler()), ('classifier', gb) ])

# Set kfold amount for cross-validation
nFold = 10
gCV = GridSearchCV( estimator=pipe, param_grid=params, cv=nFold, n_jobs=12, refit='F1',
                    return_train_score=True)

gCV.fit(X_train, Y_train)
