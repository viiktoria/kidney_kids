
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def heat_map(X):
    f,ax = plt.subplots(figsize=(12, 12))
    return sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


def rfe(estimator, step, cv, scoring, X, y):
    '''this function select the optimal number of features to be selected given an estimator which not KNeighborsClassifier
       '''

    # fitting the estimator/model
    rfecv = RFECV(estimator=estimator, step=step, cv=cv,scoring=scoring).fit(X, y)

    n = rfecv.n_features_                                       # optimal number of features
    best_features = X.columns[rfecv.support_]                   # the actual best features
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score of number of selected features")
    plt.plot(range(1, rfecv.cv_results_['mean_test_score'].shape[0] +1) , rfecv.cv_results_['mean_test_score'])

    return plt.figure()

def fi_selectbest(X,y,p):
    '''This function plot the p best features using the selectKbest approach'''

    # apply SelectKBest class to extract top n best features
    best_RF = SelectKBest(score_func=chi2, k=p).fit(X,y)
    dfscores = pd.DataFrame(best_RF.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)

    feat_importances_SB = pd.Series(best_RF.scores_, index=X.columns)
    plt.xlabel("features")
    plt.ylabel("scores")
    return feat_importances_SB.nlargest(p).plot.bar(x='features', y='scores')

def fi_rfe(estimator, n_features_to_select, step, X,y):
    ''' this function ranks the features. The feature ranked first (rank = 1) is the more important'''

    rfe = RFE(estimator=estimator, n_features_to_select= n_features_to_select, step=step).fit(X, y)
    dfrank = pd.DataFrame(rfe.ranking_)
    dfcolumns = pd.DataFrame(X.columns)

    #concat two dataframes for better visualization
    featurerank = pd.concat([dfcolumns,dfrank],axis=1)

    feat_importances_rfe = pd.DataFrame(rfe.ranking_, index=X.columns)
    feat_importances_rfe.columns =['rank']
    feat_importances_rfe = feat_importances_rfe.sort_values(by = ['rank'])
    feat_importances_rfe['rank'].squeeze()                 # get the dataframe back to a series
    #feat_importances_rfe.nsmallest(n_features_to_select, columns ='rank').plot.bar( y= "rank")
    return feat_importances_rfe.plot.bar( y= "rank")
