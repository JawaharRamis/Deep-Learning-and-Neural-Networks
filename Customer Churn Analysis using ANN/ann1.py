# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:41:05 2020

@author: Rishan
"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()#Creating dummy vars since country has 3 values

X=X[:,1:] #Removing one dummy var
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling- Compulsory for NN to ease calculations.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Creating the ANN

#Importing KEras and required packages

import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN

classifier=Sequential()

#Adding input layer and first hidden layer with dropout

classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
classifier.add(Dropout(p=0.1))

#Adding the second hidden layer with dropout

classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
classifier.add(Dropout(p=0.1))

#Adding the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Ftting the ANN to the training set

classifier.fit(X_train,y_train,batch_size=10,epochs=100)

#PRedcting the test set results

y_pred=classifier.predict(X_test)


# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred=(y_pred>0.5) #Converting probabilities to true or false values. if ypred >0.5, True else false
#this is done so as to obtain the confusion matrix

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""Use our ANN model to predict if the customer with the following informations will leave the bank: 

Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
So should we say goodbye to that customer ?
"""

new_prediction=classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction=(new_prediction>0.5)


#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#The kerasclassifier expects a function. so we define a fn as a classifier

def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
#Global classifier function
classifier= KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100) 
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=1)


mean=accuracies.mean()
variance=accuracies.std()

#Improving the ANN

#Dropout regulariztion to reduce overfitting if needed
from keras.layers import Dropout


#TUning the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
#The kerasclassifier expects a function. so we define a fn as a classifier

def build_classifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
#Global classifier function
classifier= KerasClassifier(build_fn=build_classifier) 

parameters= {'batch_size':[25,32],
             'epochs':[100,500],
             'optimizer':['adam','rmsprop',]}

grid_search= GridSearchCV(estimator=classifier,
                          param_grid=parameters,
                          scoring='accuracy',
                          cv=10)

grid_search=grid_search.fit(X_train,y_train)


best_parameters=grid_search.best_params_  
best_accuracy=grid_search.best_score_

#Not running now as it will take a loooooooong time

