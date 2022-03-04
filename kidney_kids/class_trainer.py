
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

#get data
X_train, X_test, y_train, y_test = get_cleaned_data()

#instantiate models:
#knn_model = Knn().return_trained_model(X_train, y_train)
log_model = LogReg().return_trained_model(X_train, y_train)
forest_model = RandomForest().return_trained_model(X_train, y_train)


#predict:
def predict(model, X_test):
    #print(X_test)
    y_predict = model.predict(X_test)
    print(y_predict)

if __name__ == '__name__':
    #print(X_test)
    print(forest_model)
    #predict(forest_model, X_test)
