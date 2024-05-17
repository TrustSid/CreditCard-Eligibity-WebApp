from flask import Flask, jsonify, request, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import stat
from scipy.stats import iqr

# Data wrangling
import pandas as pd
import numpy as np
import missingno
from collections import Counter

from sklearn import model_selection   

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score

import re

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
# Metrics for measuring our fit
from sklearn.metrics import mean_squared_error, accuracy_score

# Data visualisation
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier

# Machine learning models
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# Model evaluation
from sklearn.model_selection import cross_val_score

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier

    # Load the dataset
train = pd.read_csv('dataset.csv')
train['Gender'] = train['Gender'].replace({1: 'Male', 0: 'Female'})
train.drop(['DriversLicense'],1,inplace=True)
train.drop(['ZipCode'],1,inplace=True)
train.drop(['Gender'],1,inplace=True)
def detect_outliers(df, n, features):
    """"
    This function will loop through a list of features and detect outliers in each one of those features. In each
    loop, a data point is deemed an outlier if it is less than the first quartile minus the outlier step or exceeds
    third quartile plus the outlier step. The outlier step is defined as 1.5 times the interquartile range. Once the 
    outliers have been determined for one feature, their indices will be stored in a list before proceeding to the next
    feature and the process repeats until the very last feature is completed. Finally, using the list with outlier 
    indices, we will count the frequencies of the index numbers and return them if their frequency exceeds n times.    
    """
    outlier_indices = [] 
    for col in features: 
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR 
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col) 
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(key for key, value in outlier_indices.items() if value > n) 
    return multiple_outliers

outliers_to_drop = detect_outliers(train, 2, ['Age','Debt','Income','YearsEmployed','CreditScore',])

train.loc[outliers_to_drop, :]

train = train.drop(outliers_to_drop, axis = 0).reset_index(drop = True)



train = pd.get_dummies(train, columns = ['Ethnicity'])
train = pd.get_dummies(train, columns = ['Citizen'])
train.drop(['Industry'],1,inplace=True)
train.sort_index(axis=1, inplace=True)

train = train.astype({'Approved': int})
train = train.astype({'Age': int})
train = train.astype({'YearsEmployed': int})
train = train.astype({'PriorDefault': int})
train = train.astype({'Employed': int})
train = train.astype({'CreditScore': int})
train = train.astype({'Income': int})
train = train.astype({'Debt': int})
train = train.astype({'Married': int})
train = train.astype({'BankCustomer': int})

# Prepare the data for training
x = train.iloc[:, train.columns != 'Approved']
y = train.iloc[:, train.columns == 'Approved']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)
x_train = x_train.values
x_test = x_test.values
y_train = np.squeeze(y_train.values)
y_test = np.squeeze(y_test.values)


# Train the machine learning model
model = LogisticRegression(solver='liblinear', random_state=123)
clf = model.fit(x_train, y_train)

import joblib
joblib.dump(model, 'model.pkl')

