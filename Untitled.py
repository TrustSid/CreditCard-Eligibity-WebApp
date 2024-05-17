#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from catboost import CatBoostClassifier

# Model evaluation
from sklearn.model_selection import cross_val_score

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier

# Remove warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_csv("dataset.csv")


# In[3]:


train.head()


# In[4]:


#missing data in training set
train.isnull().sum().sort_values(ascending = False)


# In[5]:


#summary of general data distribution of training set

train.describe()


# In[6]:


train['Gender'] = train['Gender'].replace({1: 'Male', 0: 'Female'})


# In[7]:


# Value counts of the sex column

train['Gender'].value_counts(dropna = False)


# In[8]:


train.head()


# In[9]:


train[['Gender', 'Approved']].groupby('Gender', as_index = False).mean().sort_values(by = 'Approved', ascending = False)


# In[10]:


train[['BankCustomer', 'Approved']].groupby('BankCustomer', as_index = False).mean().sort_values(by = 'Approved', ascending = False)


# In[11]:


train[['Employed', 'Approved']].groupby('Employed', as_index = False).mean().sort_values(by = 'Approved', ascending = False)


# In[12]:


train[['PriorDefault', 'Approved']].groupby('PriorDefault', as_index = False).mean().sort_values(by = 'Approved', ascending = False)


# In[13]:


train[['Married', 'Approved']].groupby('Married', as_index = False).mean().sort_values(by = 'Approved', ascending = False)


# In[14]:


mean_age = train[train['Approved'] == 1]['Age'].mean()


# In[15]:


print(mean_age)


# In[16]:


mean_age_notapp = train[train['Approved'] == 0]['Age'].mean()
print(mean_age_notapp)


# In[17]:


# Value counts of the Pclass column 

train['Ethnicity'].value_counts(dropna = False)


# In[18]:


train[['Ethnicity', 'Approved']].groupby('Ethnicity', as_index = False).mean().sort_values(by = 'Approved', ascending = False)


# In[19]:


train[['DriversLicense', 'Approved']].groupby('DriversLicense', as_index = False).mean().sort_values(by = 'Approved', ascending = False)


# In[20]:


# Value counts of the Pclass column 

train['Citizen'].value_counts(dropna = False)


# In[21]:


# Value counts of the Pclass column 

train['Industry'].value_counts(dropna = False)


# In[22]:


train[['Industry', 'Approved']].groupby('Industry', as_index = False).mean().sort_values(by = 'Approved', ascending = False)


# In[23]:


mean_debt = train[train['Approved'] == 1]['Debt'].mean()
print(mean_debt)


# In[24]:


mean_debt_no = train[train['Approved'] == 0]['Debt'].mean()
print(mean_debt_no)


# In[25]:


mean_emp_yrs = train[train['Approved'] == 1]['YearsEmployed'].mean()
print(mean_emp_yrs)


# In[26]:


mean_emp_yrs_no = train[train['Approved'] == 0]['YearsEmployed'].mean()
print(mean_emp_yrs_no)


# In[27]:


category=['Gender','Married','BankCustomer','Industry','Ethnicity','PriorDefault','Employed','DriversLicense','Citizen','ZipCode','Approved']


# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import colorama
from colorama import Fore

for i in category:
    print(i+":")
    plt.figure(figsize=(20,10))
    sns.countplot(x=i,data=train,hue='Approved')
    plt.show()
    a=np.array(pd.crosstab(train.Approved,train[i]))
    (stats,p,dof,_)=chi2_contingency(a,correction=False)
    if p>0.05:
        print(Fore.RED +"'{}' is a 'bad Predictor'".format(i))
        print("p_val = {}\n".format(p))
    else:
        print(Fore.GREEN +"'{}' is a 'Good Predictor'".format(i))
        print("p_val = {}\n".format(p))
  


# In[29]:


train.drop(['DriversLicense'],1,inplace=True)


# In[30]:


train.drop(['Gender'],1,inplace=True)


# In[31]:


train.head()


# In[32]:


column_names = train.columns.tolist()
continious=list(set(column_names)-set(category))


# In[33]:


continious


# In[34]:


for i in continious:
    print(i+":")
    plt.figure(figsize=(20,10))
    sns.histplot(train[i])
    plt.xlabel(i)
    plt.ylabel('counts')
    plt.title('histogram of '+i)
    plt.show()
    train[i].plot.box(vert=False,patch_artist=True)
    plt.xlabel(i)
    plt.show()


# In[35]:


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
print("We will drop these {} indices: ".format(len(outliers_to_drop)), outliers_to_drop)


# In[36]:


# Outliers in numerical variables

train.loc[outliers_to_drop, :]


# In[37]:


# Drop outliers and reset index

print("Before: {} rows".format(len(train)))
train = train.drop(outliers_to_drop, axis = 0).reset_index(drop = True)
print("After: {} rows".format(len(train)))


# In[38]:


for i in continious:
    print(i+":")
    sns.histplot(x=train[i], hue=train.Approved)
    plt.xlabel(i)
    plt.ylabel('counts')
    plt.title('histogram of '+i)
    plt.show()
    sns.boxplot(y=train[i],x=train.Approved)
    #plt.xlabel(i)
    plt.show()


# In[39]:


train.head()


# In[40]:


train[['Citizen', 'Approved']].groupby('Citizen', as_index = False).mean().sort_values(by = 'Approved', ascending = False)


# In[41]:


from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
import statsmodels.api as smf
import colorama
from colorama import Fore

for i in continious:
    print("-------------------------------------------------------------------------------------------------------------------")
    print(i+":\n")
    print('ANOVA:\n')
    mod=ols(i+'~Approved',data=train).fit()
    aov_table=smf.stats.anova_lm(mod,type = 2)
    print(aov_table,'\n')
    print('Pvalue={}\n'.format(aov_table['PR(>F)'][0]))
    p=aov_table['PR(>F)'][0]

    if p>0.05:
        print(Fore.RED +"'{}' is a 'bad Predictor'\n".format(i))
        print('Avg of this feature is same for both card approved group and not approved group\n')
        print("p_val = {}\n".format(p))
    else:
        print('TUKEY:\n')
        print(Fore.RED +"'{}' is a 'good Predictor'\n".format(i))
        print('Avg of this feature is not same for both card approved group and not approved group\n')
        print('we need to perform Tuckey as atleast one category is different\n')
        print(Fore.GREEN +"'{}' is a 'good Predictor'\n".format(i))
        tukey=pairwise_tukeyhsd(train[i],train.Approved,alpha=0.05)
        print(tukey,'\n')


# In[42]:


train.drop(['ZipCode'],1,inplace=True)


# In[43]:


sns.heatmap(train[['Approved', 'Age', 'Income', 'YearsEmployed', 'Debt','BankCustomer','Employed','CreditScore','PriorDefault','Married']].corr(), annot = True, fmt = '.2f', cmap = 'coolwarm')


# In[44]:


train = pd.get_dummies(train, columns = ['Industry'])


# In[45]:


train.head()


# In[46]:


train = pd.get_dummies(train, columns = ['Ethnicity'])


# In[47]:


train = pd.get_dummies(train, columns = ['Citizen'])


# In[48]:


train.head()


# In[49]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[50]:


x = train[['Age','Debt','YearsEmployed','CreditScore','Income']]


# In[51]:


vif_data = pd.DataFrame()


# In[52]:


vif_data['features'] = x.columns


# In[53]:


vif_data['VIF'] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]


# In[54]:


vif_data


# In[57]:


x = train.iloc[:, train.columns != 'Approved']


# In[58]:


y = train.iloc[:, train.columns == 'Approved']


# In[59]:


from sklearn.model_selection import train_test_split


# In[60]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 123)


# In[61]:


from sklearn.linear_model import LogisticRegression


# In[62]:


model = LogisticRegression(solver = 'liblinear', random_state = 123)


# In[63]:


clf = model.fit(x_train,y_train)


# In[64]:


y_pred = clf.predict(x_test)


# In[65]:


clf.score(x_test,y_test)


# In[66]:


clf.intercept_


# In[67]:


clf.coef_


# In[68]:


y_pred


# In[ ]:




