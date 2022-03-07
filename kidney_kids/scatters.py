import seaborn as sns
from kidney_kids.data import get_cleaned_data, get_imputed_data, get_preproc_data


def scatter(feat1, feat2):

    X_train = get_cleaned_data()[0]
    df = get_imputed_data(X_train)
    target = get_cleaned_data()[2]

    feat_1 = df[f'{feat1}']
    feat_2 = df[f'{feat2}']

    return sns.scatterplot(data=df, x=feat_1, y=feat_2, hue=target)


def scatter_preproc(feat1, feat2):

    X_train = get_cleaned_data()[0]
    df = get_preproc_data(X_train)
    target = get_cleaned_data()[2]

    feat_1 = df[f'{feat1}']
    feat_2 = df[f'{feat2}']

    return sns.scatterplot(data=df, x=feat_1, y=feat_2, hue=target)
