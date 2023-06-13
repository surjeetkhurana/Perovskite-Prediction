# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 09:08:23 2022

@author: Surjeet
"""



import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

import matplotlib.pylab as plt
from sklearn.feature_selection import VarianceThreshold
from statistics import mean


#from tensorflow.keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

data = pd.read_excel("Stability_Training_Dataset.xlsx")
data = data.dropna()





X= data





targets = data["if"]


X = X.drop("mat", axis = 1)
X = X.drop("stability", axis = 1)
X = X.drop("delta_e", axis = 1)
X = X.drop("mili", axis = 1)
X = X.drop("if", axis = 1)






num_folds = 10
'''
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
inputs= sc.fit_transform(X)
'''
inputs=pd.DataFrame(X)

#inputs = pd.read_excel("RegressionNN.xlsx") 
"""
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
inputs= pd.DataFrame(sc.fit_transform(inputs))
"""
"""

var_thres = VarianceThreshold(threshold=0.01)
var_thres.fit(inputs)
var_thres.get_support()
constant_columns = [column for column in inputs.columns if column not in inputs.columns[var_thres.get_support()]]
After_Variance = inputs.drop(constant_columns,axis=1)


#PC
import matplotlib.pyplot as plt 
import seaborn as sns
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[j]  # getting the name of column
                col_corr.add(colname)
    af_corr = dataset.drop(col_corr,axis=1)
    return af_corr

af_both = correlation(After_Variance, .9)

inputs = af_both
"""

NewInput = np.empty([])



# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True , random_state = 1)  


inputs = inputs.reset_index(drop=True)

"""
rf = RandomForestClassifier(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(inputs, targets)

rf_random.best_params_


# manually inset the parameters for Random Forest Returned after Random Search

"""
accuracy_per_fold = []
Fscore_per_fold = []
precision_per_fold = []
recall_per_fold = []

df = []
df1 = []
# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
    
    print(inputs.iloc[train].shape)
    print(inputs.iloc[test].shape)
    print(targets.iloc[train].shape)
    print(targets.iloc[test].shape)
    
    #inputs[test] =  inputs[train].drop("Index", axis = 1)

    model = RandomForestClassifier(n_estimators = 200, min_samples_split= 2, min_samples_leaf=1, max_features = 'sqrt', max_depth= 50, bootstrap = True  ) 
    
    model.fit(inputs.iloc[train], targets.iloc[train])
    
    y_pred = model.predict(inputs.iloc[test])
    
    from sklearn.metrics import accuracy_score
    rs = accuracy_score
    accuracy = rs(targets.iloc[test], y_pred)
    
    from sklearn.metrics import f1_score
    fs = f1_score
    f1 = fs(targets.iloc[test], y_pred)
    
    
    
    
    
    
    from sklearn.metrics import confusion_matrix, precision_score
    precision = precision_score(targets.iloc[test], y_pred)
    
    
    from sklearn.metrics import confusion_matrix, recall_score
    recall = recall_score(targets.iloc[test], y_pred) # or optionally tp / (tp + fn)
    
    
    
    print("Fold Number = ", fold_no )
    print("Accuracy =", accuracy)
    
   
    one = np.ravel(targets.iloc[test])
    two = np.ravel(y_pred)
    df.insert(1,one)
    df1.insert(1,two)
    
    
    
    
    accuracy_per_fold.append(accuracy)
   
    Fscore_per_fold.append(f1)
    
    precision_per_fold.append(precision)
    
    recall_per_fold.append(recall)
    
    
    
    
    # Increase fold number
    fold_no = fold_no + 1    
    
print(mean(accuracy_per_fold))

dff = pd.DataFrame(df)
writer = pd.ExcelWriter('RFTest2.xlsx', engine='xlsxwriter')
dff.to_excel(writer, sheet_name='RandomForest', index=False)
writer.save()

dff1 = pd.DataFrame(df1)
writer = pd.ExcelWriter('RFPred2.xlsx', engine='xlsxwriter')
dff1.to_excel(writer, sheet_name='RandomForest', index=False)
writer.save()
   
