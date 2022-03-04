####imports####
from cmath import log
from webbrowser import get

#from logisticregression import LogReg, log_model
#from knn import Knn
#from randomforest import RandomForest

import pandas as pd
from data import get_cleaned_data

#import for pipe
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from models import log_model, knn_model, forest_model

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
X_train, X_test, y_train, y_test = get_cleaned_data()


#model instanziation
'''code for class-distribution:
knn_model = models.knn_model(X,y)
log_model = models.log_model(X,y)
forest_model = models.forest_model(X,y)'''


# creating feat_lists for pipeline:
feat_binary = X_train.columns[X_train.nunique()==2]
feat_ordered = ['sg', 'al', 'su']
feat_standard_scaling = ['hemo', 'age']
#all continous values without hemo and age as they are scaled differently
feat_continuous = [i for i in list(X_train.columns[X_train.nunique()>6]) if i not in ['hemo', 'age']]



## pipeline
def make_pipe(X, model):
    '''preprocesses the X, returns preproccessed X'''
    if model != 'knn_model':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    ordered_transformer = Pipeline([
                                ('cat_imputer', SimpleImputer(strategy='most_frequent')),
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

    X_proc = preproc_pipe.fit_transform(X)
    return X_proc

if __name__ == '__main__':
    #calling the preprocessing pipeline and instanziate+train the model wirth grid search
    X_proc = make_pipe(X_train, 'knn_model')
    model = knn_model(X_proc,y_train)[0]

    #prediction
    y_predict = model.predict(make_pipe(X_test,'knn_model'))
    print(y_predict)
