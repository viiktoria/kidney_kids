########### importing libraries ###############
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests


################################################
############    STREAMLIT    ###################
################################################

image1 = 'https://storage.googleapis.com/kidney_disaese/images/kidney.png'

st.image(image1, use_column_width=True)

st.write("""
         # Chronic Kidney Disease (CKD) Web App

         **Hey Doc!** You wanna use the robot for helping with the diagnosis?

        This website gives an overview to better understand and use **Machine Learning to predict CKD**.


         * **Data source**: kidney_desease.csv

         ***

         """)

st.header('Statistics')

##########################
###### SCATTERPLOTS ######
##########################
url = 'https://kidneykidsgoesworld-f77cyo2fpq-ew.a.run.app/scatter'
st.write("""

         ***Don't hesitate to select your features to get insight from your data, Doc!***



         """)

scatter_features_1 = ["Don't be shy, Doc .. miauuw", 'age', 'bgr', 'hemo', 'pcv']
scatter_features_2 = ["Come on! Now the second one!", 'age', 'bgr', 'hemo', 'pcv']

select_1 = st.selectbox('', options=scatter_features_1)
select_2 = st.selectbox('', options=scatter_features_2)

def scatter_plot(X1, X2, target):
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6))
    ax = sns.scatterplot(data=df, x=X1, y=X2, hue=target)
    plt.legend(labels=["chronic","non-chronic"])
    st.pyplot(fig)
    return st.write(f'{df.shape[0]} patients')


if select_1 != scatter_features_1[0] and select_2 != scatter_features_2[0]:
        st.write("""
                 ## ***BOOM!***
                 """)
        #url = 'http://127.0.0.1:8000/scatter'
        params = {'feat_1': select_1, 'feat_2': select_2}
        result = requests.get(url, params) # first endpoint
        df = pd.read_json(result.json())
        scatter_plot(df[select_1], df[select_2], df['class'])
else:
    '***Make your choice! Now!***'

st.write("""

         ***

         """)

####################
###### MODELS ######
####################

# Header
st.header('Models')

image2 = 'https://storage.googleapis.com/kidney_disaese/images/Models.png'
st.image(image2, use_column_width=True)


st.write("""
         ### Important info for you Doc!

         """)

with st.expander('KNN'):
    st.write("""
        **Definition**: supervised learning algortihm used to classify data points (CKD patients) based on the points that are most similar to it (nearest neighbours)

         **Hyperparameters considered**:\n
         ***n_neighbors***, int, default=5. Number of neighbors to use by default for kneighbors queries.\n
         ***p***, int, default=2. Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

         """)

with st.expander('Random Forest'):
    st.write("""
        **Definition**: Supervised Machine Learning that builds decision trees and takes their majority vote for classification of CKD

         **Hyperparameters considered**:\n
         ***max_depth***, int, default=None. The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.\n
         ***n_estimators***, int, default=100. The number of trees in the forest.

         """)

with st.expander('Logistic Regression'):
    st.write("""
         **Definition**: supervised learning classification algorithm used to predict the probability of CKD

         **Hyperparameter considered**:
         ***penalty***, default=l2. Specify the norm of the penalty: 'none': no penalty is added; 'l2': add a L2 penalty term and it is the default choice; 'l1': add a L1 penalty term; 'elasticnet': both L1 and L2 penalty terms are added.

         """)




### second endpoint ###
#####  User select model and hyperparameters  ##################



st.write("""

         ### Time to play around with your model!

         """)

classifier_name = st.selectbox("", ("KNN", "Random Forest", "Logistic Regression"))


def add_parameters_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        n_neighbors = st.slider("n_neighbors", 5, 15)
        p = st.selectbox("p", options=[1,2])

        params['model'] = 'knn'
        params["param1"] = n_neighbors
        params["param2"] = p
    elif clf_name == "Random Forest":
        max_depth = st.slider("max_depth", 2, 15)
        n_estimators = st.slider("n_estimators", 1, 100)

        params['model'] = 'randomforest'
        params["param1"] = max_depth
        params["param2"] = n_estimators
    else:
        penalty = st.selectbox("Penalty", ('l2', 'none'))
        c = st.selectbox("Strongness of Penalty:", (0.001, 0.1, 1, 10))

        params['model'] = 'logreg'
        params["param1"] = penalty
        params ['param2'] = c
    return params


params = add_parameters_ui(classifier_name)

url = 'https://kidneykidsgoesworld-f77cyo2fpq-ew.a.run.app/model'
#url = 'http://127.0.0.1:8000/model'

result = requests.get(url, params)

cm = result.json()

cm_array = np.array([[int(cm['cm1']), int(cm['cm2'])], [int(cm['cm3']), int(cm['cm4'])]])


def confusion_plot(cm, ac):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax = sns.heatmap(cm,annot=True,fmt="d", cmap="Blues");
    ax.set_title('Confusion Matrix');
    ax.set_xlabel('Predicted Labels');ax.set_ylabel('True Labels');
    ax.xaxis.set_ticklabels(['non-chronic', 'chronic'], fontsize=5);
    ax.yaxis.set_ticklabels(['non-chronic', 'chronic'], fontsize=5);
    st.pyplot(fig);
    #ac = round(float(ac),5)
    # st.write(f'## The Recall is {ac} ##');
    return

confusion_plot(cm_array, cm['ac'])

rounded_ = round(float(cm['ac']), 2)

if st.button("Checkout the recall"):
    st.write(f"#### the recall is {rounded_} ####")



st.write("""

         ***

         """)

####################
### DOC'S CHOICE ####
####################

# Header
st.sidebar.header('Make your choice Doc!')

st.sidebar.write("""

        What do you think, which features are important? Choose your features!


         """)

# st.info('This is a purely informational message')

age=st.sidebar.slider('age',min_value=2, max_value=90, value=54)
sg=st.sidebar.slider('urin specific gravity (sg)',min_value=1.005,max_value=1.025,step=0.005, value=1.02)
al=st.sidebar.slider('albumin (al)', min_value=0,max_value=5)
su=st.sidebar.slider('sugar (su)', min_value=0,max_value=5, value=0)
bgr=st.sidebar.slider('blood gluco random (mgs/dl)', min_value=70, max_value=500, value=129)
hemo=st.sidebar.slider('hemoglobin (gms)', min_value=3.1, max_value=17.8, step=0.1, value=12.6)
pcv=st.sidebar.slider('packed cell count (pcv)', min_value=16, max_value=55, value=39)
htn=st.sidebar.selectbox('hypertension (htn): yes (1) no (0)', options=[0,1])
dm=st.sidebar.selectbox('diabetes mellitus (dm): yes (1) no (0)', options=[0,1])
pe=st.sidebar.selectbox('pedal edema (pe): yes (1) no (0)', options=[0,1])

if st.sidebar.button("Predict now"):

    selected_features = {'age':age,'sg':sg,'su':su,'ba':0,'bgr':bgr,'hemo':hemo,'pcv':pcv,'wc':8430,'htn':htn,'pe':pe, 'bp': 80, 'bu': 44, 'sc':1.2, 'sod':136, 'pot':4, 'rc':5.2, 'al':al, 'rbc':0, 'pc':0, 'pcc':0, 'dm':dm, 'cad': 0, 'appet': 2, 'ane':0}
    url = 'https://kidneykidsgoesworld-f77cyo2fpq-ew.a.run.app/predict'

    result = requests.get(url, selected_features)


    #outputs prediction and probabilty of prediction from current local api:
    proba = round(float(result.json()['proba']), 2)
    if int(result.json()['result']) == 1:
        st.sidebar.success(f"### Your patient is at risk of CKD with a probability of {proba} ###")
    else:
        proba_non_chronic = 1-proba
        st.sidebar.success(f'### Your patient is not at risk of CKD with a probability of {proba_non_chronic} ###')

        # st.sidebar.error("Do you really, really, wanna do this?")

# age=st.slider('age',min_value=2, max_value=90, value=54)
# sg=st.slider('urin specific gravity (sg)',min_value=1.005,max_value=1.025,step=0.005, value=1.02)
# al=st.slider('albumin (al)', min_value=0,max_value=5)
# su=st.slider('sugar (su)', min_value=0,max_value=5, value=0)
# bgr=st.slider('blood gluco random (mgs/dl)', min_value=70, max_value=500, value=129)
# hemo=st.slider('hemoglobin (gms)', min_value=3.1, max_value=17.8, step=0.1, value=12.6)
# pcv=st.slider('packed cell count (pcv)', min_value=16, max_value=55, value=39)
# htn=st.selectbox('hypertension (htn): yes (1) no (0)', options=[0,1])
# dm=st.selectbox('diabetes mellitus (dm): yes (1) no (0)', options=[0,1])
# pe=st.selectbox('pedal edema (pe): yes (1) no (0)', options=[0,1])


# selected_features = {'age':age,'sg':sg,'su':su,'ba':0,'bgr':bgr,'hemo':hemo,'pcv':pcv,'wc':8430,'htn':htn,'pe':pe, 'bp': 80, 'bu': 44, 'sc':1.2, 'sod':136, 'pot':4, 'rc':5.2, 'al':al, 'rbc':0, 'pc':0, 'pcc':0, 'dm':dm, 'cad': 0, 'appet': 2, 'ane':0}


# ### third endpoint ###
# #url = 'http://127.0.0.1:8000/predict'
# url = 'https://kidneykidsgoesworld-f77cyo2fpq-ew.a.run.app/predict'

# result = requests.get(url, selected_features)


# #outputs prediction and probabilty of prediction from current local api:
# proba = round(float(result.json()['proba']), 2)
# if int(result.json()['result']) == 1:
#     st.success(f"### Your patient is at risk of CKD with a probability of {proba} ###")
# else:
#     proba_non_chronic = 1-proba
#     st.success(f'### Your patient is not at risk of CKD with a probability of {proba_non_chronic} ###')


### presenting result from old api (testimage1) ###
#url = 'https://testimage2-f77cyo2fpq-ew.a.run.app/predict'
#outputs prediction and probabilty of prediction from current image:
#for i, j in result.json()['proba'].items():
#    proba = round(j, 5)
#if int(result.json()['result']) == 1:
#    st.write(f"### Your patient is at risk with a probability of {proba} ###")
#else:
#    proba_non_chronic = 1-proba
#    st.write(f'### Your patient is not at risk with a probability of {proba_non_chronic} ###')


# Buttons
if st.button('About Us'):
    st.write('Kidney Kids 2022: Viktoria von Laer, Jeanne Mbebi, Markus Kramer, Cristian Jeraldo')
    image3 = 'https://storage.googleapis.com/kidney_disaese/images/lewagon.png'
    st.image(image3)





# {'age': 55.0,
#  'bp': 80.0,
#  'bgr': 116.5,
#  'bu': 44.0,
#  'sc': 1.2,
#  'sod': 136.0,
#  'pot': 4.0,
#  'hemo': 13.5,
#  'pcv': 41.0,
#  'wc': 6900.0,
#  'rc': 5.2,
#  'sg': 1.02,
#  'al': 0.0,
#  'su': 0.0,
#  'rbc': 0.0,
#  'pc': 0.0,
#  'pcc': 0.0,
#  'ba': 0.0,
#  'htn': 0.0,
#  'dm': 0.0,
#  'cad': 0.0,
#  'appet': 2.0,
#  'pe': 0.0,
#  'ane': 0.0}

# # Selected variables
# {'age': 55.0,
#  'bgr': 116.5,
#  'hemo': 13.5,
#  'pcv': 41.0,
#  'wc': 6900.0,
#  'sg': 1.02,
#  'su': 0.0,
#  'ba': 0.0,
#  'htn': 0.0,
#  'pe': 0.0}

# # Numeric selected variables
# {'age': 55.0,
#  'bgr': 116.5,
#  'hemo': 13.5,
#  'pcv': 41.0,
#  'wc': 6900.0}
