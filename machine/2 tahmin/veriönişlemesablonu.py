# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:43:23 2024

@author: Casper
"""

import pandas as pd
import matplotlib.pyplot as plt

data = [
    [8,  19671.5],
    [10, 23102.5],
    [11, 18865.5],
    [13, 21762.5],
    [14, 19945.5],
    [19, 28321],
    [19, 30075],
    [20, 27222.5],
    [20, 32222.5],
    [24, 28594.5],
    [25, 31609],
    [25, 27897],
    [25, 28478.5],
    [26, 28540.5],
    [29, 30555.5],
    [31, 33969],
    [32, 33014.5],
    [34, 41544],
    [37, 40681.5],
    [37, 46971],
    [42, 45869],
    [44, 49136.5],
    [49, 50651],
    [50, 56906],
    [54, 54715.5],
    [55, 52791],
    [59, 58484.5],
    [59, 56317.5],
    [65, 60936]
]

columns = ["ay", "para"]
df = pd.DataFrame(data, columns=columns)
print(df)


aylar = df[['ay']]
print(aylar)

satislar = df[["para"]]
print(satislar)




from sklearn.model_selection import train_test_split

x_train,xtest ,y_train,ytest= train_test_split(aylar,satislar,test_size = 0.33 , random_state=0)

"""
##öznitelik ölçeklendirme # farklı dünyalardaki veriler artık aynı dünyada
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

xx_train = sc.fit_transform(x_train)
xxtest = sc.fit_transform(xtest)

yy_train = sc.fit_transform(y_train )
yytest = sc.fit_transform(ytest)
  """

##model inşaası
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)
xpre = lr.predict(xtest)

x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train,y_train)
plt.plot(xtest,lr.predict(xtest))

plt.title("ay-satıs")
plt.xlabel("aylar")
plt.ylabel("satis")
    