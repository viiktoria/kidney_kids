
from logisticregression import LogReg
from knn import Knn
from randomforest import RandomForest

import pandas as pd
from data import get_cleaned_data

#import for pipe
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

#import to upload model to gcp
from gcp import storage_upload
import joblib



def initiate():
    #get data
    X_train, X_test, y_train, y_test = get_cleaned_data()

    #instantiate models:
    #knn_model = Knn().return_trained_model(X_train, y_train)
    #log = LogReg()
    #log_model = log.return_trained_model(X_train, y_train)

    forest = RandomForest()
    forest_model = forest.return_trained_model(X_train, y_train)

    y_predict = forest_model.predict(forest.preproc(X_test))
    #print(y_predict)
    return forest_model, X_test


#predict:
#def predict(model, X_test, forest):
#    #print(X_test)
#    y_predict = model.predict(forest.preproc(X_test))
#    print(y_predict)

if __name__ == '__main__':
    '''instantiate the model, train it with grid search and predict'''
    forest_model, X_test = initiate()

    """Save the model into a .joblib format"""
    joblib.dump(forest_model, 'model.joblib')
    storage_upload('forest_model')
