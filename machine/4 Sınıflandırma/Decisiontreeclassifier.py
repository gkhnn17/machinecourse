# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:17:41 2024

@author: dumlu
"""

import pandas as pd 
import  numpy as np
import matplotlib.pyplot as plt


# Verileri bir liste olarak tanımlayalım

veriler = pd.read_csv('boy-kilo.csv')

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,-1:].values.ravel()




####veri kümesi eğitme
#verilerin ölceklenmesi
from sklearn.model_selection import train_test_split


x_train,x_test ,y_train,y_test= train_test_split(x,y,test_size = 0.33 , random_state=0)


##öznitelik ölçeklendirme # farklı dünyalardaki veriler artık aynı dünyada
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

xx_train = sc.fit_transform(x_train)
xx_test = sc.transform(x_test)

###LOGİSTİC REGRESSİON
from sklearn.linear_model import LogisticRegression
log_reg =LogisticRegression(random_state=0)
log_reg.fit(xx_train,y_train)#eğitim

y_pred =log_reg.predict(xx_test)#tahmin
print(y_pred)
print(y_test)

###CONFUSİON METRİX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Logistic",cm)

###KNN
#en yakın 3 noktaya göre
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors= 5 , metric = "minkowski")
knn.fit(xx_train,y_train)

y_pred = knn.predict(xx_test)
cm = confusion_matrix(y_test, y_pred)
print("KNN",cm)

###SVM
#Grup verktörü çizer
from sklearn.svm import SVC
svc = SVC(kernel = "poly")
svc.fit(xx_train, y_train)

y_pred = svc.predict(xx_test)
cm = confusion_matrix(y_test, y_pred)
print("SVC",cm) 

###NAİVE BAYES
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(xx_train,y_train)

y_pred = gnb.predict(xx_test)
cm = confusion_matrix(y_test, y_pred)
print("GNB",cm)

###DECİSİON METRİKS
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion= 'entropy')

dtc.fit(xx_train,y_train)
y_pred = dtc.predict(xx_test)

y_pred = svc.predict(xx_test)
cm = confusion_matrix(y_test, y_pred)
print("DTC",cm)


###RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators= 10, criterion='entropy')
rfc.fit(xx_train,y_train)

y_pred = rfc.predict(xx_test)
y_proba = rfc.predict_proba(xx_test)


gnb = GaussianNB()
gnb.fit(xx_train,y_train)

y_pred = gnb.predict(xx_test)

cm = confusion_matrix(y_test, y_pred)
print("RFC",cm)

###FPR TRP THRESHOLD
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba[:,0], pos_label='e')
print(y_proba)
print(y_test)

print(fpr)
print(tpr)


