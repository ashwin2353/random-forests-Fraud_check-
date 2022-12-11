# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 13:48:38 2022

@author: ashwi
"""
import pandas as pd
df = pd.read_csv("Fraud_check.csv")

df.shape
df.dtypes
df.head(10)

df['Taxable.Income'].value_counts()
df['Taxable.Income'].max()

#===================================================
# box plot

import matplotlib.pyplot as plt
plt.boxplot(df['City.Population'],vert=False)
import numpy as np
Q1 = np.percentile(df['City.Population'],25)
Q2 = np.percentile(df['City.Population'],50)
Q3 = np.percentile(df['City.Population'],75)
IQR = Q3 -Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df['City.Population'] < LW) | (df['City.Population'] > UW)]

len(df[(df['City.Population'] < LW) | (df['City.Population'] > UW)])
# 0 out layers
#===================================================
import matplotlib.pyplot as plt
plt.boxplot(df['Work.Experience'],vert=False)
import numpy as np
Q1 = np.percentile(df['Work.Experience'],25)
Q2 = np.percentile(df['Work.Experience'],50)
Q3 = np.percentile(df['Work.Experience'],75)
IQR = Q3 -Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df['Work.Experience'] < LW) | (df['Work.Experience'] > UW)]

len(df[(df['Work.Experience'] < LW) | (df['Work.Experience'] > UW)])
# 0 out layers
#====================================================

plt.boxplot(df['Taxable.Income'],vert=False)

Q1 = np.percentile(df['Taxable.Income'],25)
Q2 = np.percentile(df['Taxable.Income'],50)
Q3 = np.percentile(df['Taxable.Income'],75)
IQR = Q3 -Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df['Taxable.Income'] < LW) | (df['Taxable.Income'] > UW)]
len(df[(df['Taxable.Income'] < LW) | (df['Taxable.Income'] > UW)])

# 0 out layers

#================================================
# convertid numarical varialble into categorical variable
pd.cut(df['Taxable.Income'], bins=[0,30000,99619], labels=("Risky","Good")).head(30)
df['Taxable.Income'] = pd.cut(df['Taxable.Income'], bins=[0,30000,99619], labels=("Risky","Good"))

df['Taxable.Income']
df["Taxable.Income"].value_counts()
df.dtypes
#==================================================
# Label encoding

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['Undergrad'] = LE.fit_transform(df['Undergrad'])
df['Marital.Status'] = LE.fit_transform(df['Marital.Status'])
df['Urban'] = LE.fit_transform(df['Urban'])

list(df)
df.info()

#==================================================
# deviding the variables into X and Y 

Y = df['Taxable.Income']
X = df[['Undergrad','Marital.Status','City.Population','Work.Experience','Urban']]

#==================================================
# Data partision
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)
 
X_train.shape
X_test.shape
Y_train.shape
Y_test .shape
#=================================================
# Model fitting
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(max_depth=12)
DTC.fit(X_train,Y_train)

Y_pred_train = DTC.predict(X_train)
Y_pred_test = DTC.predict(X_test)

DTC.tree_.max_depth
DTC.tree_.node_count

# Metrics

from sklearn.metrics import accuracy_score
print("Training accuracy", accuracy_score(Y_train,Y_pred_train).round(3))
print("Testing accuracy", accuracy_score(Y_test,Y_pred_test).round(3))

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(Y_train, Y_pred_train)
cm1
cm2 = confusion_matrix(Y_test, Y_pred_test)
cm2

#=================================================

# Tree visualization
# pip install graphviz

from sklearn import tree
import graphviz

dot_data = tree.export_graphviz(DTC,out_file=None, filled=True, rounded=True, special_characters=True)

graph = graphviz.source(dot_data)
graph

#=================================================
# Model fitting with entropy criterion
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(max_depth=12,criterion="entropy")
DTC.fit(X_train,Y_train)

Y_pred_train = DTC.predict(X_train)
Y_pred_test = DTC.predict(X_test)

DTC.tree_.max_depth
DTC.tree_.node_count

# Metrics

from sklearn.metrics import accuracy_score
print("Training accuracy", accuracy_score(Y_train,Y_pred_train).round(3))
print("Testing accuracy", accuracy_score(Y_test,Y_pred_test).round(3))

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(Y_train, Y_pred_train)
cm1
cm2 = confusion_matrix(Y_test, Y_pred_test)
cm2


#================================================
# Random forest classifier

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100,max_features=0.6,max_depth=12)
RFC.fit(X,Y)
Y_pred = RFC.predict(X)

# Metrics
from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy :", accuracy_score(Y, Y_pred))

cm3 =  confusion_matrix(Y, Y_pred)
cm3
#====================================================
# Bagging

from sklearn.ensemble import BaggingClassifier
DTC = DecisionTreeClassifier(max_depth=12)
bag = BaggingClassifier(n_estimators=100,base_estimator=(DTC),max_samples=0.6,)
bag.fit(X,Y)
Y_pred = bag.predict(X)

# Metrics
from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy :", accuracy_score(Y, Y_pred))

cm4 =  confusion_matrix(Y, Y_pred)
cm4
#====================================================
# GridsearchCV

from sklearn.model_selection import GridSearchCV
import numpy as np
RFC.get_params().keys()
d1 = {"max_features":np.arange(0.1,1,0.1), "n_estimators": range(1,500,100),"max_depth":range(1,16,1)}
grid_search = GridSearchCV(estimator = RFC, param_grid = d1, scoring="accuracy",cv=10)
grid_search = grid_search.fit(X,Y_pred)

accuracy = grid_search.best_score_
accuracy

grid_search.best_params_


# Random forest classifier

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=401,max_features=0.3,max_depth=7)
RFC.fit(X,Y)
Y_pred = RFC.predict(X)

# Metrics
from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy :", accuracy_score(Y, Y_pred))

cm3 =  confusion_matrix(Y, Y_pred)
cm3

#================================================================
# Gradient boosting 
from sklearn.ensemble import GradientBoostingClassifier
Gboost = GradientBoostingClassifier(learning_rate=0.4,n_estimators=100)
Gboost.fit(X,Y)
Y_pred = Gboost.predict(X)

from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy :", accuracy_score(Y, Y_pred))

cm3 =  confusion_matrix(Y, Y_pred)
cm3

#================================================================
# Adaboosting
from sklearn.ensemble import AdaBoostClassifier
Adaboost = AdaBoostClassifier(base_estimator=DTC, n_estimators=100,learning_rate=0.001)
Adaboost.fit(X,Y)
Y_pred = Adaboost.predict(X)


from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy :", accuracy_score(Y, Y_pred))

cm3 =  confusion_matrix(Y, Y_pred)
cm3
#=================================================================
# comparing to all ensemble methods Adaboosting is giving the best accuracy score






