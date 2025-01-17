# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:13:46 2024

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

from sklearn.linear_model import LogisticRegression
log_reg =LogisticRegression(random_state=0)
log_reg.fit(xx_train,y_train)

y_pred =log_reg.predict(xx_test)
print(y_pred)
print(y_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Logistic",cm)

#KNN
#en yakın 3 noktaya göre
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors= 5 , metric = "minkowski")
knn.fit(xx_train,y_train)

y_pred = knn.predict(xx_test)
cm = confusion_matrix(y_test, y_pred)
print("KNN",cm)

#SVM
#Grup verktörü çizer
from sklearn.svm import SVC
svc = SVC(kernel = "poly")
svc.fit(xx_train, y_train)

y_pred = svc.predict(xx_test)
cm = confusion_matrix(y_test, y_pred)
print("SVC",cm) 


