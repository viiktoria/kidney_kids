from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import GridSearchCV


class Knn():
    def __init__(self):
        pass


    def scaling(self,X,y):
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        return X


    def make_model(self,X,y):
        X = self.scaling(X)

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
