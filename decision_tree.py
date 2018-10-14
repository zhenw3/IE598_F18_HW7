# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 01:05:03 2018

@author: zhenw
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

data=pd.read_csv('wine.csv')
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.1, stratify=y,random_state=4)

params=np.arange(10,201,10)
score=[]
score1=[]

for i in params:
    dt=RandomForestClassifier(criterion='gini', 
                              max_depth=4, n_estimators=i,random_state=21)
    dt.fit(X_train,y_train)
    score.append(np.mean(cross_val_score(dt,X_train,y_train,cv=10)))
    score1.append(dt.score(X_test,y_test))

# Create a scatter plot with train and test actual vs predictions
plt.scatter(params, np.array(score1)-0.0001, label='test')
plt.scatter(params, np.array(score), label='train')
plt.xlabel("n_estimator")
plt.ylabel(("accuracy"))
plt.legend()
plt.show()

best_index=np.argsort(np.array(score)+np.array(score1))[-1]
best_param=params[best_index]

dt=RandomForestClassifier(criterion='gini', 
                              max_depth=4, n_estimators=best_param,random_state=21)
dt.fit(X,y)

# Get feature importances from our random forest model
feature_names=X.columns
importances = dt.feature_importances_

# Get the index of importances from greatest importance to least
sorted_index = np.argsort(importances)[::-1]
x = range(len(importances))

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            X.columns[sorted_index[f]], 
                            importances[sorted_index[f]]))

# Create tick labels 
labels = np.array(feature_names)[sorted_index]
plt.bar(x, importances[sorted_index], tick_label=labels)

# Rotate tick labels to vertical
plt.xticks(rotation=90)
plt.show()

print("My name is Zhen Wang")
print("My NetID is: zhenw3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
