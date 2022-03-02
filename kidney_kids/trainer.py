####imports####
from cmath import log
from webbrowser import get

from logisticregression import LogReg, log_model
from knn import Knn
from randomforest import RandomForest

import pandas as pd
from data import get_clean_data

#import for pipe
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

import models

'''used scaling methods:
logreg:
['age'] -> standard
['bp'] -> minmax
['bgr'] -> minmax
['bu'] -> minmax
...
['hemo'] -> standard

'''
'''knn:
all -> minmax
'''
'''tree:
['age'] -> standard
['bp'] -> minmax
['bgr'] -> minmax
['bu'] -> minmax
...
['hemo'] -> standard
    '''
###baselinescore
baseline = 250/400

####get the data####
X,y = get_clean_data()


#model instanziation
knn_model = models.knn_model(X,y)
log_model = models.log_model(X,y)
forest_model = models.forest_model(X,y)

###make pipeline including the model
def make_pipe(model):

    if model != knn_model:
        feat_binary = X.columns[X.nunique()==2]
        feat_ordered = ['sg', 'al', 'su']
        feat_standard_scaling = ['hemo', 'age']

        #all continous values without hemo and age as they are scaled differently
        feat_continuous = [i for i in list(X.columns[X.nunique()>6]) if i not in ['hemo', 'age']]

        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    ordered_transformer = Pipeline([
                                ('cat_imputer', SimpleImputer(strategy='most_frequent'))
                                ('mm_scaler', MinMaxScaler())
                                ])

    binary_transformer = Pipeline([
                                ('cat_imputer', SimpleImputer(strategy='most_frequent'))
                                ])

    cont_transformer = Pipeline([
                                ('num_imputer', SimpleImputer()),
                                ('mm_scaler', MinMaxScaler())
                                ])
    standard_transformer = Pipeline([
                                    ('num_imputer', SimpleImputer()),
                                    ('scaler', scaler)
                                    ])

    preproc_pipe = ColumnTransformer([
                                        ('ord_trans', ordered_transformer, feat_ordered),
                                        ('bin_trans', binary_transformer, feat_binary),
                                        ('cont_trans', cont_transformer, feat_continuous),
                                        ('stand_trans', standard_transformer, feat_standard_scaling)
                                     ])

    final_pipe = Pipeline([
                        ('pipe', preproc_pipe),
                        ('model', model)
                        ])
    return final_pipe
