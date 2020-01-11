# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 17:16:10 2020

@author: user
"""
#Support Vector MAchines
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#Load dataset
data = pd.read_csv('E:\\heart_disease_prediction\\dataset\\heart.csv')
print(data.head())
print(data.columns)
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,fmt='.1f')
plt.show()

#test train split
x = data.iloc[:,:-1]
y = data.iloc[:,13]
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print('X_train',X_train.shape)
print('X_test',X_test.shape) 
print('y_train',y_train.shape)
print('y_test',y_test.shape)

#Normalization as the first process
# Normalize
X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train)).values
X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values


clf = svm.SVC()
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print('Accurancy :',accuracy_score(y_test, y_pred))
print("KNN TRAIN score with ",format(clf.score(X_train, y_train)))

cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix")
sns.heatmap(cm,annot=True)
plt.show()


