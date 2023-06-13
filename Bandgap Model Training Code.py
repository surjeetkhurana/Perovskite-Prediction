# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 23:50:18 2023

@author: Surjeet
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 23:11:22 2023

@author: Surjeet
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 00:14:15 2022

@author: Surjeet
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 05:07:19 2022

@author: Surjeet
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 22:51:11 2022

@author: Surjeet
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

#from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score


from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.feature_selection import VarianceThreshold


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

data = pd.read_csv("Bandgap_Training_Dataset.csv")
data = data.dropna()





dfObj = data

# Get names of indexes for which column bandgap has value 0
indexNames = dfObj[ dfObj['bandgap'] == 0 ].index
 


# Delete these row indexes from dataFrame
dfObj.drop(indexNames , inplace=True)

targets  = data["bandgap"]




X = dfObj.drop("bandgap", axis = 1)


num_folds = 10
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

af_both = correlation(After_Variance, .8)

inputs = af_both

"""
"""
rf = RandomForestRegressor(random_state = 42)
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
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(inputs, targets)

rf_random.best_params_
"""

#manually insert the new variables in the model and retrain   

#model = RandomForestRegressor(n_estimators = 1600, min_samples_split = 5, min_samples_leaf =  1, max_features = 'sqrt', max_depth = 70, bootstrap = False )
model = RandomForestRegressor(n_estimators = 1600, min_samples_split = 5, min_samples_leaf =  1, max_features = 'sqrt', max_depth = 70, bootstrap = False )
NewInput = np.empty([])
model1 = RandomForestRegressor()
Rsquare_per_fold = []
Mse_per_fold = []
Mae_per_fold = []

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True , random_state = 1)  


inputs = inputs.reset_index(drop=True)

df = []
df1 = []
# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
    
    print(inputs.iloc[train].shape)
    print(inputs.iloc[test].shape)
    print(targets[train].shape)
    print(targets[test].shape)
    
    #inputs[test] =  inputs[train].drop("Index", axis = 1)

     
    
    model1.fit(inputs.iloc[train], targets[train])
    
    y_pred = model1.predict(inputs.iloc[test])
    
    from sklearn.metrics import r2_score
    rs = r2_score
    Rsquare = rs(targets[test], y_pred)
    
    from sklearn.metrics import mean_absolute_error
    rs = mean_absolute_error
    mae = rs(targets[test], y_pred)

    from sklearn.metrics import mean_squared_error
    rs = mean_squared_error
    mse = rs(targets[test], y_pred)
        
    
    
    print("Fold Number = ", fold_no )
    print("R_sruared =", Rsquare)
    
   
    one = np.ravel(targets[test])
    two = np.ravel(y_pred)
    df.insert(1,one)
    df1.insert(1,two)
    
    
    
    
    Rsquare_per_fold.append(Rsquare)
    Mse_per_fold.append(mse)
    Mae_per_fold.append(mae)
    
    
    
    
    
    
    
    # Increase fold number
    fold_no = fold_no + 1    
    
dff = pd.DataFrame(df)
writer = pd.ExcelWriter('RFTestFinal.xlsx', engine='xlsxwriter')
dff.to_excel(writer, sheet_name='RandomForest', index=False)
writer.save()

dff1 = pd.DataFrame(df1)
writer = pd.ExcelWriter('RFPredFinal.xlsx', engine='xlsxwriter')
dff1.to_excel(writer, sheet_name='RandomForest', index=False)
writer.save()



