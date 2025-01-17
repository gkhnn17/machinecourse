# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:57:31 2024

@author: Casper
"""
"""
Spyder Editor

This is a temporary script file.
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

# DataFrame'i görelim

boy = df[['boy']]

print(boy)

#eksik veriler

from sklearn.impute import SimpleImputer


imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

yas = df.iloc[:,1:4].values
imputer = imputer.fit(yas[:, 0:3])

# Transform the data
yas[:, 0:3] = imputer.transform(yas[:, 0:3])
print(yas)




