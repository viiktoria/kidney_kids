from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd

#import for pipe
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

class LogReg:
    def __init__(self):
        pass

    def preproc(self,X_train):

        # creating feat_lists for pipeline
        feat_binary = X_train.columns[X_train.nunique()==2]
        feat_ordered = ['sg', 'al', 'su']
        feat_standard_scaling = ['hemo', 'age']
        #all continous values without hemo and age as they are scaled differently
        feat_continuous = [i for i in list(X_train.columns[X_train.nunique()>6]) if i not in ['hemo', 'age']]


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
                                        ('scaler', StandardScaler())
                                        ])

        preproc_pipe = ColumnTransformer([
                                            ('ord_trans', ordered_transformer, feat_ordered),
                                            ('bin_trans', binary_transformer, feat_binary),
                                            ('cont_trans', cont_transformer, feat_continuous),
                                            ('stand_trans', standard_transformer, feat_standard_scaling)
                                        ])

        X_proc = preproc_pipe.fit_transform(X_train)
        return X_proc

    def log_model(self,X_proc,y):
        model = LogisticRegression()
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

        # define search space
        space = dict()
        space['solver'] = ['newton-cg', 'lbfgs', 'saga', 'sag', 'liblinear']
        space['penalty'] = ['none', 'l2', 'l1', 'elastcinet']
        space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
        space['max_iter'] = [50,100,200,500]

        # define search
        search = GridSearchCV(model, space, scoring='recall', n_jobs=-1, cv=cv)
        result = search.fit(X_proc,y)

        model = result.best_estimator_

        return model

    def return_trained_model(self,X_train,y_train):
        X_proc = self.preproc(X_train)
        model = self.log_model(X_proc,y_train)

        return model

    def custom_model(self, penalty,C, X_train,y_train):
        custom_model = LogisticRegression(penalty,C)
        custom_model = custom_model.fit(X_train,y_train)
        return custom_model
