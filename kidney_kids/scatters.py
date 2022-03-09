import seaborn as sns
from io import BytesIO
from kidney_kids.data import get_cleaned_data, get_imputed_data, get_preproc_data
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import recall_score


def scatter(feat1, feat2):

    df = get_cleaned_data()[0]
    target = get_cleaned_data()[2]


    #mit den zwei zeilen drüber fkt die api noch nicht, mit den folgenden gibt es an dieser stelle keinen fehler, deshalb hier hinzugefügt
    #feat_1 = df[feat1]
    #feat_2 = df[feat2]

    # scatterplot muss irgendwie zu image convertiert werden:
    #plot = sns.scatterplot(data=df, x=feat_1, y=feat_2, hue=target)
    #buf = BytesIO()
    #plot.savefig(buf, format="png")
    #buf.seek(0)
    #return buf

    return sns.scatterplot(data=df, x=feat1, y=feat2, hue=target)


def scatter_preproc(feat1, feat2):

    X_train = get_cleaned_data()[0]
    df = get_preproc_data(X_train)
    target = get_cleaned_data()[2]
    graph = sns.scatterplot(data=df, x=feat1, y=feat2, hue=target)

    return graph


def plot_df(feat1, feat2):
    X_train = get_cleaned_data()[0]
    df = get_imputed_data(X_train)
    df['class'] = get_cleaned_data()[2]


    return df[[feat1, feat2, 'class']]


def confusion_score(classifier):
    '''this function plots the confusion matrix given a classifier'''

    X_train, X_test, y_train, y_test = get_cleaned_data()
    X_train_preproc = get_preproc_data(X_train)
    X_test_preproc = get_preproc_data(X_test)


    clr_rf = classifier.fit(X_train_preproc,y_train)

    ac = recall_score(y_test,classifier.predict(X_test_preproc))
    cm = confusion_matrix(y_test,clr_rf.predict(X_test_preproc))

    #return(sns.heatmap(cm,annot=True,fmt="d"), f'Accuracy is {ac}')
    return ac, cm[0][0], cm[0][1], cm[1][0], cm[1][1]
