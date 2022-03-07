from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns

def confusion_score(classifier, X_train,y_train, y_test,X_test):
    '''this function plots the confusion matrix given a classifier'''
    clr_rf = classifier.fit(X_train,y_train)

    ac = accuracy_score(y_test,classifier.predict(X_test))
    cm = confusion_matrix(y_test,clr_rf.predict(X_test))

    return(sns.heatmap(cm,annot=True,fmt="d"), f'Accuracy is {ac}')
