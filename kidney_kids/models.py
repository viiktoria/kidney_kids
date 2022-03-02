from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold


def log_model(X,y):
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
    result = search.fit(X,y)

    df = pd.DataFrame(result.cv_results_)

    model = result.best_estimator_

    return [model, df, result.best_params_]


'''old code when scaling was still part of the classes
def scaling(self,X,y):
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    return X'''

def knn_model(self,X,y):
    #X = self.scaling(X)

    knn_model = KNeighborsClassifier()
    params = {'n_neighbors': np.arange(2,11),
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], #there are more metrices
            'weights': ['uniform', 'distance']
            }


    search = GridSearchCV(knn_model, param_grid=params, scoring='recall')
    result = search.fit(X,y)
    df = pd.DataFrame(result.cv_results_)
    return [result.best_estimator_, df, result.best_params_]

def forest_model(X,y):
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
