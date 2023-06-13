# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 22:00:17 2023

@author: Surjeet
"""


"""
"""
import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score


from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.feature_selection import VarianceThreshold




data = pd.read_csv("1-ABX3.csv")
#data = data.dropna()


X = data.drop("Count", axis = 1)
X = X.drop("Material", axis = 1)




'''
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
inputs= sc.fit_transform(X)
'''
inputs=pd.DataFrame(X)


"""
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
inputs= pd.DataFrame(sc.fit_transform(inputs))
"""





import pickle

loaded_model = pickle.load(open("FinalStabilityPred.sav", 'rb'))
result = loaded_model.predict(inputs)
print(result)

