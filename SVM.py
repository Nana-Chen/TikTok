#!use/bin/env python
# -*- coding:utf-8 _*-
"""
@author : chenmeiyi
@file : FCM.py
@time : 2020/06/09
@desc : 
"""
import copy
import math
import random
import time
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('  ',header=0,delim_whitespace=True)
array = data.values
X=array[:,0:-1]
Y=array[:,-1]
validation_size=0.2
seed = 7
X_train,X_validation,Y_train,Y_validation = train_test_split(X,Y,test_size=validation_size,random_state=seed)

scaler = StandardScaler().fit(X_train)
rescaledX=scaler.transform(X_train)
param_grid = {'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21]}
model = KNeighborsRegressor()
kfold = KFold(n_splits=10,random_state=7)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring='neg_mean_squared_error',cv=kfold)
grid_result = grid.fit(X=rescaledX, Y=Y_train)
print('Best：%s 使用%s' %(grid_result.best_score_,grid_result.best_params_))
cv_results = zip(grid_result.cv_results_['mean_test_score'],grid_result.cv_results_['std_test_score'],grid_result.cv_results_['params'])
for mean,std,param in cv_results:
    print('%f(%f) with %r' % (mean,std,param) )


num_folds =10
seed = 7
kfold = KFold(n_splits=num_folds,random_state=seed)
clf = svm.SVC()
clf.fit(X,Y_train,sample_weight=None)
result = clf.predict(test_data)
print(result)
