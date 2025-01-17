

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    ["us", 167, 62, 40, "k"],
    ["fr", 174, 70, 47, "e"],
    ["fr", 194, 90, 23, "e"], 
    ["fr", 187, 80, 27, "e"],
    ["fr", 183, 88, 28, "e"],
    ["fr", 159, 80, 29, "k"],
    ["fr", 164, 66, 32, "k"],
    ["fr", 166, 56, 42, "k"]
]

# Sütun adlarını tanımlayalım
columns = ["ulke", "boy", "kilo", "yas", "cinsiyet"]

# Verileri DataFrame'e dönüştürelim
df = pd.DataFrame(data, columns=columns)

boy =df[['boy']]



## encoder ordinaldan > numerice
ulke = df.iloc[:,0:1].values
print(ulke)



from sklearn import preprocessing
##ulkeler 1-2-0
le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(df.iloc[:,0])


#onehot clumn başlıkları ülkeler kendi artık bir tabloda
"""[0. 0. 1.]
   [0. 0. 1.]
   [0. 1. 0.]
   [1. 0. 0.]"""
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()


#cinsiyet
c = df.iloc[:,-1:].values
print(c)



from sklearn import preprocessing
##ulkeler 1-2-0
le = preprocessing.LabelEncoder()

c[:,0] = le.fit_transform(df.iloc[:,-1])

print(c)

#onehot clumn başlıkları ülkeler kendi artık bir tabloda
"""[0. 0. 1.]
   [0. 0. 1.]
   [0. 1. 0.]
   [1. 0. 0.]"""
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)



#numpy dizileri dtaframe dönüştürülür
sonuc = pd.DataFrame(data=ulke, index = range (22), columns = ['fr', 'tr', 'us'])


sonuc2 = df[['boy','kilo','yas']]


cinsiyet = df.iloc[:,-1].values    
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1] , index = range(22), columns=['cinsiyet'])

#data fram birleştirme işlemi
s1 = pd.concat([sonuc, sonuc2], axis=1)
s2 = pd.concat([s1, sonuc3], axis=1)

####veri kümesi eğitme
#verilerin ölceklenmesi
from sklearn.model_selection import train_test_split


x_train,x_test ,y_train,y_test= train_test_split(s1,sonuc3,test_size = 0.33 , random_state=0)


####veri kümesi eğitme
#verilerin ölceklenmesi
from sklearn.model_selection import train_test_split


x_train,x_test ,y_train,y_test= train_test_split(s1,sonuc3,test_size = 0.33 , random_state=0)



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_predict = regressor.predict(x_test)

boy = s2.iloc[:,3:4].values

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)

x_train, x_test,y_train,y_test = train_test_split(veri, boy, test_size = 0.33 , random_state=0)

#BACKWARD ELİMİNATİON
"""
bu kod doğrusal regresyon modeli oluşturur, bu modeli eğitim verileri ile eğitir ve 
ardından test verileri üzerinde tahminler yapar. Tahmin edilen değerler y_predict değişkeninde
saklanır.
"""
##P DEGERİ
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)#xtrain bağımsız ytrain bağımlı değişken
y_predict = regressor.predict(x_test)# yeni verilere göre tahmin yapmak

#pi değerini hesaplama #significant değerini geçen
import statsmodels.api as sm
x = np.append(arr = np.ones((22,1)).astype(int), values = veri, axis=1 )# çoklu regresyondaki değişken

x_l = veri.iloc[:,[0,1,2,3]].values
x_l = np.array(x_l,dtype = float)
model = sm.OLS(boy,x_l).fit()#boy bağımlı ,x,l bapımgız değişken
print(model.summary())#p değeri en yüksek p değerliyi eliyoruz 4.,5. elendi

