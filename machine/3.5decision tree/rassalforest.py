# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:08:17 2024

@author: dumlu
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('maaslar.csv')

#data slicing
x = veriler.iloc[:,1:2].values#bağımsız
y = veriler.iloc[:,2:].values#bağımlı




#linearregression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x,y)
plt.plot(x, lin_reg.predict(x))
plt.show()

#polynomial regression
from sklearn.preprocessing  import PolynomialFeatures
poly_reg =PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

print(x_poly) 

#polinamal özelliklerden eğitmek için linear regresyon 
#fit(x): Bu metot, sadece modelin parametrelerini öğrenir ama veriyi dönüştürmez.
#transform(x): Bu metot, zaten fit edilmiş (öğrenilmiş) parametrelerle veriyi dönüştürür.

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
plt.scatter(x, y, color= "red")
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)))
plt.show()

#print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))



##öznitelik ölçeklendirme # farklı dünyalardaki veriler artık aynı dünyada
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_skate = sc1.fit_transform(x)
sc2 = StandardScaler()
y_skate = sc2.fit_transform(y).ravel()

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_skate,y_skate)

plt.scatter(x_skate,y_skate )
plt.plot(x_skate,svr_reg.predict(x_skate),color ="green")
plt.show()
print(svr_reg.predict([[6.6]]))


#DECİSİON TREE regression
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)

r_dt.fit(x,y)

z = x+0.5
k= x-0.4


plt.scatter(x, y,color = "red")
plt.plot(x,r_dt.predict(x))
plt.plot(x,r_dt.predict(z))
plt.plot(x,r_dt.predict(k))
plt.show()



print(r_dt.predict([[6.6]]))

#RandomForest regression
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)

rf_reg.fit(x,y.ravel())
print(rf_reg.predict([[6.6]]))

plt.scatter(x, y, color = 'red')
plt.plot(x,rf_reg.predict(x)  )
plt.plot(x,rf_reg.predict(z))
plt.plot(x,rf_reg.predict(k),color='yellow')


plt.show()



#r kare doğruluk oranı
from sklearn.metrics import r2_score
print("Randomforest R2' degeri","\n",r2_score(y,rf_reg.predict(x)))#(gerçek ,predict)



print("Polynomial R2' degeri","\n",r2_score(y,lin_reg2.predict(poly_reg.fit_transform(x))))











