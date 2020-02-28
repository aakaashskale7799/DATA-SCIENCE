# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:28:19 2020

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm

train=pd.read_csv(r"C:\Users\User\Desktop\DATA SCIENCE\house-prices-advanced-regression-techniques\train.csv")
test=pd.read_csv(r"C:\Users\User\Desktop\DATA SCIENCE\house-prices-advanced-regression-techniques\test.csv")
print(train.head())
train.shape
print ('The train data has {0} rows and {1} columns'.format(train.shape[0],train.shape[1]))
print ('The test data has {0} rows and {1} columns'.format(test.shape[0],test.shape[1]))
train.isnull().sum()
train.info()
train.columns[train.isnull().any()]
mean=train["LotFrontage"].mean()
train["LotFrontage"].fillna(mean)
#missing value counts in each of these columns
miss = train.isnull().sum()/len(train)
miss = miss[miss > 0]
miss.sort_values(inplace=True)
#visualising missing values
miss = miss.to_frame()
miss.columns = ['count']
miss.index.names = ['Name']
miss['Name'] = miss.index
#plot the missing value count
sns.set(style="whitegrid", color_codes=True)
sns.barplot(x="Name", y="count", data=miss)
plt.xticks(rotation = 90)
sns.plt.show()
#SalePrice
sns.distplot(train['SalePrice'])
print( "The skewness of SalePrice is {}".format(train['SalePrice'].skew()))
#now transforming the target variable
target = np.log(train['SalePrice'])
print ('Skewness is', target.skew())


