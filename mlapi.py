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
import joblib

model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/loan', methods=['POST'])
def loan():
    # Get the input data from the form

    Age = int(25)
    Married = int(0)
    Debt = int(2)
    BankCustomer = int(1)
    YearsEmployed = int(3)
    PriorDefault = int(1)
    Employed = int(1)
    CreditScore = int(request.form['CreditScore'])
    Income = int(request.form['Income'])
    Ethnicity_Asian = int(1)
    Ethnicity_Black = int(0)
    Ethnicity_Latino = int(0)
    Ethnicity_Other = int(0)
    Ethnicity_White = int(0)
    Citizen_ByBirth = int(1)
    Citizen_ByOtherMeans = int(0)
    Citizen_Temporary = int(0)

 
    

    # Convert the input data into a dict
    input_dict = {
        'Age': [Age],
        'Debt': [Debt],
        'Married': [Married],
        'BankCustomer': [BankCustomer],
        'YearsEmployed': [YearsEmployed],
        'PriorDefault': [PriorDefault],
        'Employed': [Employed],
        'Income': [Income],
        'CreditScore': [CreditScore],   
        'Ethnicity_Asian': [Ethnicity_Asian],
        'Ethnicity_Black': [Ethnicity_Black],
        'Ethnicity_Latino': [Ethnicity_Latino],
        'Ethnicity_Other': [Ethnicity_Other],
        'Ethnicity_White': [Ethnicity_White],
        'Citizen_ByBirth': [Citizen_ByBirth],
        'Citizen_ByOtherMeans': [Citizen_ByOtherMeans],
        'Citizen_Temporary': [Citizen_Temporary]
    }

    input_df = pd.DataFrame.from_dict(input_dict)[['Age', 'Debt', 'Married', 'BankCustomer', 'YearsEmployed', 
    'PriorDefault', 'Employed', 'Income', 'CreditScore', 'Ethnicity_Asian', 
    'Ethnicity_Black', 'Ethnicity_Latino', 'Ethnicity_Other', 'Ethnicity_White', 'Citizen_ByBirth',
     'Citizen_ByOtherMeans', 'Citizen_Temporary' ]]

    # Use the model to predict the loan approval
    prediction = model.predict(input_df)[0]

    # Return the prediction as a JSON object
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
      app.run(debug=True)
