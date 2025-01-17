# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:49:27 2024

@author: dumlu
"""


import pandas as pd 
import  numpy as np
import matplotlib.pyplot as plt


# Verileri bir liste olarak tanımlayalım

veriler = pd.read_excel('Iris.xls')

print(veriler)

x = veriler.iloc[:,:4].values
y = veriler.iloc[:,-1:].values.ravel()


