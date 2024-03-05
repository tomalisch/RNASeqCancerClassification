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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
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

# Split data into train, validation, and test at 80-10-10
X_train, X_test, Y_train, Y_test = train_test_split( xData, yData, test_size=0.2 )
X_test, X_val, Y_test, Y_val = train_test_split( X_test, Y_test, test_size=0.5)

# Balance classes using synthetic minority oversampling
sm = SMOTE(random_state=2)
X_train, Y_train = sm.fit_resample(X_train, Y_train)


