# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 19:14:40 2024

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

print(lin_reg.predict([[11]]))

print(lin_reg.predict([[6.6]]))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))

##SVR

sc1 = StandardScaler()
xx_train = sc.fit_transform(x_train)

xxtest = sc.fit_transform(xtest)
