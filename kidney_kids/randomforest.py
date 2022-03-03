from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd

class RandomForest():
    def __init__(self):
        pass

    def make_model(X,y):
        '''create the model, do the gridsearch
        and return fitted model with best params'''
        rfc=RandomForestClassifier()

        param_grid = {
        'n_estimators': [100, 300, 500],
        'criterion' : ['gini', 'entropy'],
        'max_depth' : [3,5,7,10,15],
        'min_samples_split' : [2, 3, 5, 7]
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)

        cv_rfc.best_params_

        search = GridSearchCV(rfc, param_grid=param_grid, scoring='recall')
        result = search.fit(X,y)

        df = pd.DataFrame(result.cv_results_)
        model = result.best_estimator_

        return [model, df, result.best_params_]
