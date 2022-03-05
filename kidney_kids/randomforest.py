from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

#import for pipe
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

class RandomForest:
    def __init__(self):
        pass

    def preproc(self, X_train):

        ''' returns preprocessed data for log reg and tree
        ( as two features are scaled differnetly then in knn )'''
        # creating feat_lists for pipeline
        feat_binary = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        feat_ordered = ['sg', 'al', 'su']
        feat_continuous = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc',
        'rc']



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


    def forest_model(self,X_proc,y_train):
        '''create the model, do the gridsearch
        and return fitted model with best params'''
        rfc=RandomForestClassifier()

        param_grid = {
        'n_estimators': [100, 300, 500],
        'criterion' : ['gini', 'entropy'],
        'max_depth' : [3,5,7,10,15],
        'min_samples_split' : [2, 3, 5, 7]
        }

        search = GridSearchCV(rfc, param_grid=param_grid, scoring='recall')
        result = search.fit(X_proc,y_train)

        #grid_search_df = pd.DataFrame(result.cv_results_)
        model = result.best_estimator_

        return model

    def return_trained_model(self,X_train,y_train):
        X_proc = self.preproc(X_train)
        model = self.forest_model(X_proc,y_train)

        return model
