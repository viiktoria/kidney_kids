#get data from csv and clean data. get_cleaned_data returns cleaned dataset with X and y
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

path = '../raw_data/kidney_disease.csv'

def get_cleaned_data(path=path):
    '''load data from csv
    and use cleaning fct to clean them'''
    df = pd.read_csv(path)
    y = df['classification']
    X = df.drop(columns= {'classification', 'id'})

    X = replacing_numerical_features(X)
    X,y = replacing_binary_features(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

    return X_train, X_test, y_train, y_test

def replacing_numerical_features(X):
    '''cleaning: strips \t at beginning of number and replaces ? with nan values'''
    X['pcv'] = X['pcv'].str.lstrip('\t')
    X['pcv'] = X['pcv'].replace(to_replace='?',value=np.nan).astype(float)
    X['wc'] = X['wc'].str.lstrip('\t')
    X['wc'] = X['wc'].replace(to_replace='?',value=np.nan).astype(float)
    X['rc'] = X['rc'].str.lstrip('\t')
    X['rc'] = X['rc'].replace(to_replace='?',value=np.nan).astype(float)

    return X

def replacing_binary_features(X, y):
    '''encoding: replacing Yes --> 1 no --> 0'''
    X[['htn','dm','cad','pe','ane']] = X[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
    X[['rbc','pc']] = X[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
    X[['pcc','ba']] = X[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
    X[['appet']] = X[['appet']].replace(to_replace={'good':2,'poor':1,'no':0})
    ## replacing t_values to 0 or 1, by assuming it s close to 0 or 1, respectively
    X['cad'] = X['cad'].replace(to_replace='\tno',value=0)
    X['dm'] = X['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1})

    #encoding the target:
    y= y.replace(to_replace={'ckd':1,'notckd':0, 'ckd\t': 1}).astype(int)

    return X, y

if __name__ == '__main__':
    X,y = get_cleaned_data("../raw_data/kidney_disease.csv")
    print(X)
    print(y)
