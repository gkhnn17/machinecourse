# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 20:38:07 2024

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
from sklearn import preprocessing

columns = ["ulke", "boy", "kilo", "yas", "cinsiyet"]

# Verileri DataFrame'e dönüştürelim
veriler = pd.DataFrame(data, columns=columns)


from sklearn.impute import SimpleImputer
imputer = SimpleImputer (missing_values=np.nan, strategy='mean')
yas=  veriler.iloc[:,1:4].values
print(yas)

imputer = imputer.fit(yas[:,0:3])
yas[:,0:3] = imputer.transform(yas[:,0:3])
print(yas)
ulke = veriler.iloc[:,0:1].values
print(ulke)

le = preprocessing.LabelEncoder()
ulke[:, 0] = le.fit_transform(veriler.iloc[:, 0].fillna('missing'))  # NaN değerleri 'missing' ile doldur
print("Label Encoded 'ulke':")
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print("One Hot Encoded 'ulke':")
print(ulke)

sonuc = pd.DataFrame(data = ulke, index = range(22))



