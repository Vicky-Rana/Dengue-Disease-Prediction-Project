# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:57:02 2017

@author: prabakap, Vicky
"""

#Importing necessary Libraies
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#Importing the dataset
train_knn_sj = pd.read_csv('train_knn_sj.csv')
test_knn_sj = pd.read_csv('test_knn_sj.csv')
train_knn_iq = pd.read_csv('train_knn_iq.csv')
test_knn_iq=pd.read_csv('test_knn_iq.csv')

dengu_dep = pd.read_csv('dengue_labels_train.csv')
dengu_dep_sj = dengu_dep[dengu_dep['city'] == 'sj']
dengu_dep_iq = dengu_dep[dengu_dep['city'] == 'iq']

dengu_dep_sj = dengu_dep_sj.iloc[:,3].values
dengu_dep_iq = dengu_dep_iq.iloc[:,3].values

from sklearn.model_selection import train_test_split
X_train_sj, X_test_sj, y_train_sj, y_test_sj= train_test_split(train_knn_sj,dengu_dep_sj,test_size=0.2, random_state=42)
#Convert Integer to float for normalization
y_train_sj=y_train_sj.astype(float)
y_test_sj=y_test_sj.astype(float)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
#Standization for Independent Variables of sj
std_sj=StandardScaler()
X_train_sj=std_sj.fit_transform(train_knn_sj)
X_test_sj=std_sj.transform(test_knn_sj)

#Standardization for Dependent variables of sj
std_sj_y=StandardScaler()
y_train_sj=std_sj_y.fit_transform(dengu_dep_sj)

#Standization for Independent Variables of iq
std_iq=StandardScaler()
X_train_iq=std_iq.fit_transform(train_knn_iq)
X_test_iq=std_iq.transform(test_knn_iq)

#Standardization for Dependent variables of iq
std_iq_y=StandardScaler()
y_train_iq=std_iq_y.fit_transform(dengu_dep_iq)

#Linear Regression Model
from sklearn.linear_model import LinearRegression
regress=LinearRegression()
regress.fit(X_train_sj,y_train_sj)
#list(zip(colnam, regress.coef_))
regress.score(X_train_sj,y_train_sj)
#Prediction Part
case_pred_sj_lr=regress.predict(X_test_sj)
case_pred_sj_lr=std_sj_y.inverse_transform(case_pred_sj_lr)

regress_iq=LinearRegression()
regress_iq.fit(X_train_iq,y_train_iq)
#list(zip(colnam, regress.coef_))
regress_iq.score(X_train_iq,y_train_iq)
#Prediction Part
case_pred_iq_lr=regress.predict(X_test_iq)
case_pred_iq_lr=std_sj_y.inverse_transform(case_pred_iq_lr)

#Random Forest Regression model
from sklearn.ensemble import RandomForestRegressor
regr_rf = RandomForestRegressor(max_depth=3, random_state=42)
regr_rf.fit(X_train_sj, y_train_sj)
case_pred_sj_rf=regr_rf.predict(X_test_sj)
case_pred_sj_rf=std_sj_y.inverse_transform(case_pred_sj_rf)

regr_rf_iq = RandomForestRegressor(max_depth=3, random_state=42)
regr_rf_iq.fit(X_train_iq, y_train_iq)
case_pred_iq_rf=regr_rf.predict(X_test_iq)
case_pred_iq_rf=std_sj_y.inverse_transform(case_pred_iq_rf)

#SVM Regression Model
from sklearn import svm
svm_regression_model = svm.SVR(kernel='poly')
svm_regression_model.fit(X_train_sj,y_train_sj)
case_pred_sj_svm=svm_regression_model.predict(X_test_sj)
case_pred_sj_svm=std_sj_y.inverse_transform(case_pred_sj_svm)

svm_regression_model_iq = svm.SVR(kernel='poly')
svm_regression_model_iq.fit(X_train_iq,y_train_iq)
case_pred_iq_svm=svm_regression_model_iq.predict(X_test_iq)
case_pred_iq_svm=std_iq_y.inverse_transform(case_pred_iq_svm)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
Regressor = Sequential()

# Adding the input layer and the first hidden layer
Regressor.add(Dense(units = 47, kernel_initializer = 'uniform', activation = 'relu', input_dim = 97))
#hidden units=6 and 'uniform' weight initialization

# Adding the second hidden layer
Regressor.add(Dense(units = 47, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the second hidden layer
#Regressor.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
Regressor.add(Dense(units = 1, kernel_initializer = 'uniform'))

# Compiling the ANN
Regressor.compile(optimizer = 'adam', loss = 'mean_absolute_error')

# Fitting the ANN to the Training set
Regressor.fit(X_train_sj, y_train_sj, batch_size = 10, epochs = 200)

# Predicting the Test set results
case_pred_sj_ann = Regressor.predict(X_test_sj)
case_pred_sj_ann=std_sj_y.inverse_transform(case_pred_sj_ann)

#case_pred_sj[case_pred_sj<=0]=0

# Initialising the ANN
Regressor_iq = Sequential()

# Adding the input layer and the first hidden layer
Regressor_iq.add(Dense(units = 47, kernel_initializer = 'uniform', activation = 'relu', input_dim = 97))
#hidden units=6 and 'uniform' weight initialization

# Adding the second hidden layer
Regressor_iq.add(Dense(units = 47, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the second hidden layer
#Regressor.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
Regressor_iq.add(Dense(units = 1, kernel_initializer = 'uniform'))

# Compiling the ANN
Regressor_iq.compile(optimizer = 'adam', loss = 'mean_absolute_error')

# Fitting the ANN to the Training set
Regressor_iq.fit(X_train_iq, y_train_iq, batch_size = 10, epochs = 200)

# Predicting the Test set results
case_pred_iq_ann = Regressor.predict(X_test_iq)
case_pred_iq_ann=(std_iq_y.inverse_transform(case_pred_iq_ann))

#case_pred_iq[case_pred_iq<=0]=0

#case_pred_sj=np.mean(case_pred_sj_lr,case_pred_sj_rf,case_pred_sj_svm,case_pred_sj_ann)

#final_pred=pd.DataFrame(np.concatenate((case_pred_sj,case_pred_iq), axis=0))

#final_pred.to_csv('Case Prediction.csv',index=False)

#Mean Absolute Error Evaluation
#from sklearn.metrics import mean_absolute_error
#mean_absolute_error(y_test_sj, case_pred_sj)






