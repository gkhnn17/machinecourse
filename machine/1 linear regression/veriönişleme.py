# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 21:25:15 2024

@author: Casper
"""

import pandas as pd 
import  numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# Verileri bir liste olarak tanımlayalım
data = [
    ["tr", 130, 30, 10, "e"],
    ["tr", 125, 36, 11, "e"],
    ["tr", 135, 34, 10, "k"],
    ["tr", 133, 30, 9, "k"],
    ["tr", 129, 38, 12, "e"],
    ["tr", 180, 90, 30, "e"],
    ["tr", 190, 80, 25, "e"],
    ["tr", 175, 90, 35, "e"],
    ["tr", 177, 60, 22, "k"],
    ["us", 185, 105, 33, "e"],
    ["us", 165, 55, 27, "k"],
    ["us", 155, 50, 44, "k"],
    ["us", 160, 58, 39, "k"],
    ["us", 162, 59, 41, "k"],
    ["us", 167, 62, None, "k"],
    ["fr", 174, 70, 47, "e"],
    ["fr", 180,90, 23, "e"], 
    ["fr", 187, 80, 27, "e"],
    ["fr", 183, 88, 28, "e"],
    ["fr", 159, 40, 29, "k"],
    ["fr", 164, 66, 32, "k"],
    ["fr", 166, 56, 42, "k"]
]

# Sütun adlarını tanımlayalım
columns = ["ulke", "boy", "kilo", "yas", "cinsiyet"]

# Verileri DataFrame'e dönüştürelim
df = pd.DataFrame(data, columns=columns)

boy =df[['boy']]



##eksik veriler
from sklearn.impute import SimpleImputer
imputer = SimpleImputer (missing_values=np.nan, strategy='mean')
yas=  df.iloc[:,1:4].values
print(yas)

imputer = imputer.fit(yas[:,0:3])
yas[:,0:3] = imputer.transform(yas[:,0:3])


## encoder ordinaldan > numerice
ulke = df.iloc[:,0:1].values
print(ulke)



from sklearn import preprocessing
##ulkeler 1-2-0
le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(df.iloc[:,0])

print(ulke)

#onehot clumn başlıkları ülkeler kendi artık bir tabloda
"""[0. 0. 1.]
   [0. 0. 1.]
   [0. 1. 0.]
   [1. 0. 0.]"""
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


#numpy dizileri dtaframe dönüştürülür
sonuc = pd.DataFrame(data=ulke, index = range (22), columns = ['fr', 'tr', 'us'])
print(sonuc)

sonuc2 = pd.DataFrame(data = yas, index= range(22), columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet = df.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet , index = range(22), columns=['cinsiyer'])

#data fram birleştirme işlemi
s= pd.concat([sonuc,sonuc2],axis = 1)
print(s)


s2 = pd.concat([s,sonuc3],axis= 1)
print(s2)

####veri kümesi eğitme
#verilerin ölceklenmesi
from sklearn.model_selection import train_test_split


x_train,xtest ,y_train,ytest= train_test_split(s,sonuc3,test_size = 0.33 , random_state=0)


##öznitelik ölçeklendirme # farklı dünyalardaki veriler artık aynı dünyada
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

xx_train = sc.fit_transform(x_train)
xxtest = sc.fit_transform(xtest)



