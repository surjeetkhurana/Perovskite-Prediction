# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 00:28:26 2023

@author: Surjeet
"""

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




data = pd.read_csv("Validation_Dataset1_with_Features.csv")
#data = data.dropna()

bandgap = data["Bandgap"]

data = data.drop("Filename", axis = 1)
data = data.drop("Bandgap", axis = 1)
data = data.drop("Count", axis = 1)






import pickle

loaded_model = pickle.load(open("FinalRegModel.sav", 'rb'))
result = loaded_model.predict(data)
print(result)

from sklearn.metrics import r2_score
rs = r2_score
Rsquare = rs(bandgap, result)
print(Rsquare)


