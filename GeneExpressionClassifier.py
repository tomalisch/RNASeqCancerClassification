## Using public labeled gene expression data to predict cancer type using different models
# 1) Load and clean data
# 2) Balance groups and create data loaders
# 3) Build, train, validate, and test accuracy across different ML models
# 4) Print output and conclude

## Dependencies
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.decomposition import NMF, PCA
from sklearn_evaluation import plot as evalPlot
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

# Note that data are already log1p transformed
xData = pd.read_csv(filepathX)
yData = pd.read_csv(filepathY)

# Drop NaNs and redundant columns from dataframes
xData.dropna(inplace=True)
xData.drop(columns=["Unnamed: 0"], inplace=True)
yData.dropna(inplace=True)
yData.drop(columns=["Unnamed: 0"], inplace=True)

# Split data into train and test sets (80/20 split)
X_train, X_test, Y_train, Y_test = train_test_split( xData, np.ravel(yData), test_size=0.2 )

# Standard scale outside of pipeline as PCA requires it
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Perform PCA and select #PCs based on explained variance
pca = PCA()
# Determine transformed features
X_train_pca = pca.fit_transform(X_train_std)
# Determine explained variance using explained_variance_ration_ attribute
exp_var_pca = pca.explained_variance_ratio_
# Cumulative sum of eigenvalues; This will be used to create step plot
# for visualizing the variance explained by each principal component.
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
# Create the visualization plot
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Determine how many PCs are needed to reach 90% explained variance (or cap at 50 if more would be needed)
numPCs = np.where(cum_sum_eigenvalues>=0.9)[0][0] if np.where(cum_sum_eigenvalues>=0.9)[0][0]<50 else 50
print('Choosing {} PCs, reaching {} explained variance'.format(numPCs, cum_sum_eigenvalues[numPCs]))

# Set up PCA
pca = PCA(n_components=numPCs)
# Fit on the train set only
pca.fit(X_train_std)
# Apply transform to both the train set and the test set - rename for clarity later in the pipe.
X_train = pca.transform(X_train_std)
X_test = pca.transform(X_test_std)

# Initialize classifier object for parameter optimization grid search
gb = GradientBoostingClassifier(criterion='friedman_mse', init=None,
                             learning_rate=0.1, loss='log_loss', max_depth=3,
                             max_features=None, max_leaf_nodes=None,
                             min_impurity_decrease=0.0,
                             min_samples_leaf=1, min_samples_split=2,
                             min_weight_fraction_leaf=0.0, n_estimators=100,
                             n_iter_no_change=None,
                             random_state=None, subsample=1.0, tol=0.0001,
                             validation_fraction=0.2, verbose=0,
                             warm_start=False)

# Create list of tunable parameters for grid search
params = [
 {'classifier__learning_rate': (0.001, 0.01, 0.1)}, {'classifier__max_depth': (1, 5)},
 {'classifier__min_impurity_decrease': (0.0, 0.01, 0.1)},{'classifier__min_samples_leaf': (1, 2, 5)}, {'classifier__min_samples_split': (2, 5, 10)}
 , {'classifier__min_weight_fraction_leaf': (0.3, 0.15, 0.1)}, {'classifier__n_estimators': (10, 100, 200)}, {'classifier__tol': ( 0.0001, 0.001)} ]

pipe = Pipeline([ ('sampling', SMOTE()), ('classifier', gb) ])

# Set kfold amount for cross-validation, target is cancer among 5 categories - choose F1 macro score as performance metric due to imbalanced data
nFold = 10
gCV = GridSearchCV( estimator=pipe, param_grid=params, cv=nFold, scoring='f1_macro', n_jobs=12, refit=True,
                    return_train_score=True, verbose=1)

gCV.fit(X_train, Y_train)

# Visualize fit of models by hyperparameters
mean_scores = np.array(gCV.cv_results_["mean_test_score"])
mean_scores = pd.DataFrame(mean_scores)
ax = mean_scores.plot.bar()
print('Best score: {}'.format(gCV.best_score_))
print('Best parameters: {}'.format(gCV.best_params_))
print('Best model: {}'.format(gCV.best_estimator_))
###evalPlot.grid_search(gCV.cv_results_, change='n_estimators', kind='bar')


# Choose best model parameters based on F1 score and get best model performance on test set
optimizedModel = gCV.best_estimator_

# Since we use SMOTE, accuracy is still relevant as a metric
# Note that test data is already scaled and PCA'd
y_pred = optimizedModel.predict(X_test)

# Report confusion matrix
confusion_matrix = confusion_matrix(Y_test, y_pred)
print("Confusion matrix:\n{}".format(confusion_matrix))
# Classification report
labels = range(len(np.unique(Y_test)))
target_names = np.unique(Y_test)
report = classification_report(Y_test, y_pred,labels=labels,target_names=target_names,output_dict=True)
print("Classification report:\n{}".format(report))
# Plot heatmap of classification report
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)

# Since we use SMOTE, accuracy is still relevant as a metric
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)