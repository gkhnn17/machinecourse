# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 18:11:25 2024

@author: dumlu
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('odev3.csv')


#data slicing
x = veriler.iloc[:,2:3].values#bağımsız
y = veriler.iloc[:,-1:].values#bağımlı



#linearregression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x,y)
plt.plot(x, lin_reg.predict(x))
plt.show()


#r kare doğruluk oranı
from sklearn.metrics import r2_score

print("Linear R2' degeri","\n",r2_score(y,lin_reg.predict(x)))


#pi değerini hesaplama #significant değerini geçen
import statsmodels.api as sm
model = sm.OLS(lin_reg.predict(x),x)
print(model.fit().summary())



#polynomial regression
print("POLY")
from sklearn.preprocessing  import PolynomialFeatures
poly_reg =PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)


#polinamal özelliklerden eğitmek için linear regresyon 
#fit(x): Bu metot, sadece modelin parametrelerini öğrenir ama veriyi dönüştürmez.
#transform(x): Bu metot, zaten fit edilmiş (öğrenilmiş) parametrelerle veriyi dönüştürür.

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
plt.scatter(x, y, color= "red")
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)))
plt.show()

model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(x)),x)
print(model2.fit().summary())

print("Polynomial R2' degeri","\n",r2_score(y,lin_reg2.predict(poly_reg.fit_transform(x))))



#svr
print("SVR")
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

model3 = sm.OLS(svr_reg.predict(x_skate),x_skate)
print(model3.fit().summary())

print("svr R2' degeri","\n",r2_score(y_skate,svr_reg.predict(x_skate)))




#DECİSİON TREE regression
print("DECİSİONTREE")
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

model4 = sm.OLS(r_dt.predict(x),x)
print(model4.fit().summary())

print("svr R2' degeri","\n",r2_score(y,r_dt.predict(x)))



#RandomForest regression
print("RANDOMFOREST")
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)

rf_reg.fit(x,y.ravel())

plt.scatter(x, y, color = 'red')
plt.plot(x,rf_reg.predict(x)  )
plt.plot(x,rf_reg.predict(z))
plt.plot(x,rf_reg.predict(k),color='yellow')


plt.show()

model5 = sm.OLS(rf_reg.predict(x),x)
print(model5.fit().summary())

print("svr R2' degeri","\n",r2_score(y,rf_reg.predict(x)))



