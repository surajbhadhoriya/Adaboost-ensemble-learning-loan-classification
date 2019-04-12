# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:01:25 2019

@author: SURAJ BHADHORIYA
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:00:16 2019

@author: SURAJ BHADHORIYA
"""

#load libraries
import numpy as np
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.cross_validation import train_test_split
import pandas as pd
import pydotplus
from IPython.display import Image

#read data
df=pd.read_csv("loan_dataset.csv")

#print the names of col.
col=df.columns
print(col)
#make label
df['safe_loans']=df['bad_loans'].apply(lambda s:+1 if s==0 else 0)
print(df['safe_loans'])


#find the +ive & -ive % of loan
pos_loan=len(df[df['safe_loans']==1])
neg_loan=len(df[df['safe_loans']==-0])
pos=(pos_loan*100)/122607
neg=(neg_loan*100)/122607
print("positive loan %",pos)
print("negative loan %",neg)

#put all feature together
feature=['grade','term','home_ownership','emp_length']
label=['safe_loans']

#make new dataframe where only feature and label append
loan=df[feature+label]

#make one hot encoding on dataframe
loan1=pd.get_dummies(loan)

#make feature one hot encoading
x=pd.get_dummies(loan[feature])
#takindsome sampleof data
x1=x[:40000]
#make label
y=loan['safe_loans']
#taking some sample of label 
y1=y[:40000]


X_train,X_test,y_train,y_test=train_test_split(x1,y1,test_size=0.2)

#ADABOOST CLASIFIERS METHOD
from sklearn. ensemble import AdaBoostClassifier
Ab=AdaBoostClassifier(base_estimator=None,n_estimators=20)
Ab.fit(X_train,y_train)
accuracy=Ab.score(X_test,y_test)
print("accuarcy",accuracy)

#WE USE DIFFERENT LEARN OR CLASSIFIER AS PER YOUR REQURIMENT
abd=AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=20,learning_rate=1)
abd.fit(X_train,y_train)
acc=abd.score(X_test,y_test)
print("accuarcy",acc)

#*******************************************************************
#tere are lot of other algo for boosting
#XGBOOST CLASSIFIER
"""
from xgboost.sklearn import XGBClassifier
bst=XGBClassifier()
bst.fit()

#GRADIENT BOOSTING
from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
gb.fit()

"""























