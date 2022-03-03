from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd



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

if __name__ == '__main__':
    logreg = LogReg()
