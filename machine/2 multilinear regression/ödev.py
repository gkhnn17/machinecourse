# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:14:35 2024

@author: Casper
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

columns = ["outlook","temperature","humidity","windy","play"]
veriler = [
    ["sunny",85,85,"FALSE","no"],
    ["sunny",80,90,"TRUE","no"],
    ["overcast",83,86,"FALSE","yes"],
    ["rainy",70,96,"FALSE","yes"],
    ["rainy",68,80,"FALSE","yes"],
    ["rainy",65,70,"TRUE","no"],
    ["overcast",64,65,"TRUE","yes"],
    ["sunny",72,95,"FALSE","no"],
    ["sunny",69,70,"FALSE","yes"],
    ["rainy",75,80,"FALSE","yes"],
    ["sunny",75,70,"TRUE","yes"],
    ["overcast",72,90,"TRUE","yes"],
    ["overcast",81,75,"FALSE","yes"],
    ["rainy",71,91,"TRUE","no"]
    ]
# Verileri DataFrame'e dönüştürelim
veriler = pd.DataFrame(veriler, columns=columns)


from sklearn import preprocessing

veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)

c = veriler2.iloc[:,:1]

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()

havadurumu = pd.DataFrame(data= c, index = range(14),columns=['o','r','s'])
#data fram birleştirme işlemi
s1 = pd.concat([havadurumu, veriler.iloc[:,1:3]], axis=1)
s2 = pd.concat([s1,veriler2.iloc[:,-2:]],axis=1)

print(s2)
#verilerin ölceklenmesi
from sklearn.model_selection import train_test_split
x = s2.drop(columns = ['humidity'])
y = s2['humidity']
#humidty bağımlı değişkeni için diğerli bağımsız       *diğer   ,   *humidy
x_train,x_test ,y_train,y_test= train_test_split(x,y,test_size = 0.33 , random_state=0)


#predict
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_predict = regressor.predict(x_test)

#BACKWARD ELİMİNATİON

#pi değerini hesaplama #significant değerini geçen
import statsmodels.api as sm
X = np.append(arr = np.ones((14,1)).astype(int), values = x, axis=1 )# çoklu regresyondaki değişken

X_l = x.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype = float)
model = sm.OLS(y,X_l).fit()#humity bağımlı ,x,l bapımgız değişken
print(model.summary())#p değeri en yüksek p değerliyi eliyoruz 4.,5. elendi

x = x.drop(columns=["windy"])

#pi değerini hesaplama #significant değerini geçen
import statsmodels.api as sm
X = np.append(arr = np.ones((14,1)).astype(int), values = x, axis=1 )# çoklu regresyondaki değişken

X_l = x.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l,dtype = float)
model = sm.OLS(y,X_l).fit()#humity bağımlı ,x,l bapımgız değişken
print(model.summary())#p değeri en yüksek p değerliyi eliyoruz 4.,5. elendi

x_train =x_train.drop(columns=["windy"])
x_test = x_test.drop(columns=["windy"])

#windysiz 
regressor.fit(x_train,y_train)

y_predict = regressor.predict(x_test)

