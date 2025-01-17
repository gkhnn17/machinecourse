# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 20:38:30 2024

@author: dumlu
"""


import pandas as pd 
import  numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# Verileri bir liste olarak tanımlayalım

veriler = pd.read_csv('boy-kilo.csv')



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
