import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('healthcare_dataset_stroke_data.csv')

# Input DataFrame
X=df.iloc[:,1:-1]
X.head()

# Target DataFrame
Y=df.iloc[:,-1]
Y.head()

# Splitting the dataset into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Checking and imputing NaN values
for cols in X_train.columns:
    print(" NaN values in " + cols + " = " + str (X_train[cols].isnull().sum()))

median = X_train['bmi'].median()
X_train['bmi'] = X_train['bmi'].replace(np.NaN, median)

# Selecting numerical features
num_features = [features for features in X_train.columns if X_train[features].dtypes != 'O']
num_features.remove('hypertension')
num_features.remove('heart_disease')

# Selecting categorical features
cat_features = [features for features in X_train.columns if X_train[features].dtypes == 'O']
cat_features.append('hypertension')
cat_features.append('heart_disease')

# Handling the outliers using quantile-based Capping and Flooring technique
upper_limit = X_train['bmi'].quantile(0.99)
lower_limit = X_train['bmi'].quantile(0.01)
X_train['bmi'] = np.where(X_train['bmi'] >= upper_limit, upper_limit,
        np.where(X_train['bmi'] <= lower_limit, lower_limit, X_train['bmi']))

upper_limit = X_train['avg_glucose_level'].quantile(0.99)
lower_limit = X_train['avg_glucose_level'].quantile(0.01)
X_train['avg_glucose_level'] = np.where(X_train['avg_glucose_level'] >= upper_limit, upper_limit,
        np.where(X_train['avg_glucose_level'] <= lower_limit, lower_limit, X_train['avg_glucose_level']))

# Encoding the categorical features into numerical features
from sklearn.preprocessing import LabelEncoder
l1 = LabelEncoder()
l2 = LabelEncoder()
l3 = LabelEncoder()
l4 = LabelEncoder()
l5 = LabelEncoder()

X_train['gender'] = l1.fit_transform(X_train['gender'])
X_train['ever_married'] = l2.fit_transform(X_train['ever_married'])
X_train['work_type'] = l3.fit_transform(X_train['work_type'])
X_train['Residence_type'] = l4.fit_transform(X_train['Residence_type'])
X_train['smoking_status'] = l5.fit_transform(X_train['smoking_status'])

# Feature Engineering on Testing Data
# Checking and imputing NaN values
for cols in X_test.columns:
    print(" NaN values in " + cols + " = " + str (X_test[cols].isnull().sum()))

X_test['bmi'] = X_test['bmi'].replace(np.NaN, median)

# Selecting numerical features
num_features_test = [features for features in X_test.columns if X_test[features].dtypes != 'O']
num_features_test.remove('hypertension')
num_features_test.remove('heart_disease')

# Selecting categorical features
cat_features_test = [features for features in X_test.columns if X_test[features].dtypes == 'O']
cat_features_test.append('hypertension')
cat_features_test.append('heart_disease')

# Handling the outliers using quantile-based Capping and Flooring technique
upper_limit = X_test['bmi'].quantile(0.99)
lower_limit = X_test['bmi'].quantile(0.01)
X_test['bmi'] = np.where(X_test['bmi'] >= upper_limit, upper_limit,
        np.where(X_test['bmi'] <= lower_limit, lower_limit, X_test['bmi']))

upper_limit = X_test['avg_glucose_level'].quantile(0.99)
lower_limit = X_test['avg_glucose_level'].quantile(0.01)
X_test['avg_glucose_level'] = np.where(X_test['avg_glucose_level'] >= upper_limit, upper_limit,
        np.where(X_test['avg_glucose_level'] <= lower_limit, lower_limit, X_test['avg_glucose_level']))

# Encoding the categorical features into numerical features
X_test['gender'] = l1.transform(X_test['gender'])
X_test['ever_married'] = l2.transform(X_test['ever_married'])
X_test['work_type'] = l3.transform(X_test['work_type'])
X_test['Residence_type'] = l4.transform(X_test['Residence_type'])
X_test['smoking_status'] = l5.transform(X_test['smoking_status'])

# Step 5 : Feature Selection
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i,j])>threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr             

corr_features = correlation(X_train, 0.85)

# Feature Selection based on importance of the features
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X_train, Y_train)
feature_imp = model.feature_importances_

# Handling the imbalanced dataset
# Applying the SMOTE technique to handle the unbalanced dataset
from collections import Counter
print('Original dataset target feature category counter {}'.format(Counter(Y_train)))

from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_train, Y_train = sm.fit_resample(X_train,Y_train)
print('Resampled dataset target feature category counter {}'.format(Counter(Y_train)))


# Step 6 : Model Building and Evaluationv (XGBoost)
import xgboost as xgb
model_xgboost = xgb.XGBClassifier(learning_rate=0.1, max_depth=9, n_estimators=1000)

model_xgboost.fit(X_train,Y_train)

Y_pred_train_xgboost = model_xgboost.predict(X_train)
Y_pred_test_xgboost = model_xgboost.predict(X_test)

# Evaluating the XGBoost model
from sklearn.metrics import accuracy_score
print('Train accuracy: ' + str(accuracy_score(Y_train, Y_pred_train_xgboost)))
print('Test accuracy: ' + str(accuracy_score(Y_test, Y_pred_test_xgboost)))

import sys
np.set_printoptions(threshold=sys.maxsize)
print(X_test.iloc[9,:])

import pickle
pickle.dump(model_xgboost, open('xgboost.pkl', 'wb'))