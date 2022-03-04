from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd

#import for pipe
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

class Knn:
    def __init__(self):
        pass

    def preproc(self,X_train):

        # creating feat_lists for pipeline
        feat_binary = X_train.columns[X_train.nunique()==2]
        feat_ordered = ['sg', 'al', 'su']
        feat_continuous = [X_train.nunique()>6]


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

        preproc_pipe = ColumnTransformer([
                                            ('ord_trans', ordered_transformer, feat_ordered),
                                            ('bin_trans', binary_transformer, feat_binary),
                                            ('cont_trans', cont_transformer, feat_continuous)
                                        ])

        X_proc = preproc_pipe.fit_transform(X_train)
        return X_proc

    def knn_model(self,X_proc,y_train):
        knn_model = KNeighborsClassifier()
        params = {'n_neighbors': np.arange(2,11),
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], #there are more metrices
                'weights': ['uniform', 'distance']
                }


        search = GridSearchCV(knn_model, param_grid=params, scoring='recall')
        result = search.fit(X_proc,y_train)
        df = pd.DataFrame(result.cv_results_)
        return [result.best_estimator_, df, result.best_params_]

    def return_trained_model(self,X_train,y_train):
        X_proc = self.preproc(X_train)
        model = self.knn_model(X_proc,y_train)
        print(len(model))
        return model[0]
