########### importing libraries ###############
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from PIL import Image
import requests
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# for this try are used classifiers and accuracy, would not be necessary if model is taken from our modules

####### importing data and preprocessing #######
from kidney_kids.data import get_cleaned_data
from kidney_kids.randomforest import RandomForest

X_train, X_test, y_train, y_test = get_cleaned_data()



forest_model = RandomForest()
X_train_preproc = forest_model.preproc(X_train)
X_test_preproc = forest_model.preproc(X_test)

################################################
############    STREAMLIT    ###################
################################################

image1 = Image.open('kidney.png')

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

st.write("""

         ***Select features for scatterplot***

         """)
# numeric_columns = X_num_visualiation(X) # function from "data_preparation" notebook
# x_values = st.selectbox("X axis", options=numeric_columns)
# y_values = st.selectbox("Y axis", options=numeric_columns)

# x_values = st.selectbox("X axis", options=X_train_preproc)
# y_values = st.selectbox("Y axis", options=X_train_preproc)

# params = {"x": x_values, "y": y_values}

# plot=px.scatter(data_frame=X_train_preproc, x=x_values, y=y_values) # define df
# st.plotly_chart(plot)

# scatterplots = requests.get(url, params) ######### UNCOMMENT ########

st.write("""

         ***

         """)

####################
###### MODELS ######
####################

# Header
st.header('Models')

image2 = Image.open('Models.png')
st.image(image2, use_column_width=True)

st.write("""
        ### KNN
         **Definition**: supervised learning algortihm used to classify data points (CKD patients) based on the points that are most similar to it (nearest neighbours)

         **Hyperparameters considered**:\n
         ***n_neighbors***, int, default=5. Number of neighbors to use by default for kneighbors queries.\n
         ***p***, int, default=2. Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

        ### Random Forest
         **Definition**: Supervised Machine Learning that builds decision trees and takes their majority vote for classification of CKD

         **Hyperparameters considered**:\n
         ***max_depth***, int, default=None. The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.\n
         ***n_estimators***, int, default=100. The number of trees in the forest.


         ### Logistic Regression
         **Definition**: supervised learning classification algorithm used to predict the probability of CKD

         **Hyperparameter considered**:
         ***penalty***, default=l2. Specify the norm of the penalty: 'none': no penalty is added; 'l2': add a L2 penalty term and it is the default choice; 'l1': add a L1 penalty term; 'elasticnet': both L1 and L2 penalty terms are added.

         """)

#####  User select model and hyperparameters  ##################

classifier_name = st.selectbox("Select Classifier", ("KNN", "Random Forest", "Logistic Regression"))

st.write("Shape of Training Dataset", X_train_preproc.shape)

def add_parameters_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        n_neighbors = st.slider("n_neighbors", 1, 15)
        p = st.selectbox("p", options=[1,2])
        # params["model"] = clf_name
        params["n_neighbors"] = n_neighbors
        params["p"] = p
    elif clf_name == "Random Forest":
        max_depth = st.slider("max_depth", 2, 15)
        n_estimators = st.slider("n_estimators", 1, 100)
        # params["model"] = clf_name
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    else:
        penalty = st.selectbox("penalty", 'L1', 'L2', 'elastic')
        # solver = st.selectbox("solver", 'liblinear', 'lbfgs')
        # params["model"] = clf_name
        params["penalty"] = penalty
        # params["solver"] = solver
    return params

st.write(classifier_name)
add_parameters_ui(classifier_name)
# params =

# confusion_matrix = requests.get(url, params) ######### UNCOMMENT ########


# mod_params = (classifier_name, params)
# {'model': model, 'penalty': penalty...}


#### KIDNEY KIDS, FROM HERE USE THE ORIGINAL MODELS??? ################

# params = add_parameters_ui(classifier_name)

# def get_classifier(clf_name, params):
#     if clf_name == "KNN":
#         clf = KNeighborsClassifier(n_neighbors=params["n_neighbors"], p=params["p"])
#     elif clf_name == "Random Forest":
#         clf = RandomForestClassifier(max_depth=params["max_depth"], n_estimators=params["n_estimators"])
#     else:
#         clf = LogisticRegression(C=params["C"], max_iter=params["max_iter"])

#     return clf

# clf = get_classifier(classifier_name, params)

# clf.fit(X_train_preproc, y_train)

# y_pred = clf.predict(X_test_preproc)

# acc = accuracy_score(y_test, y_pred)
# st.write(f"classifier = {classifier_name}")
# st.write(f"Accuracy = {acc}")


#####  Plot actual vs predicted classification  ##################

# fig, ax = plt.subplots()
# ax.scatter(y_pred, y_test, alpha=0.8, cmap=plt.viridis)
# plt.xlabel("CKD predicted")
# plt.ylabel("Actual CKD")

# st.pyplot(fig)

st.write("""

         ***

         """)

####################
### DOC'S CHOICE ###
####################

# Header
st.header('Make your choice Doc!')

st.write("""

        What do you think, which features are important? Choose your features!


         """)

#####  User select features. Selected features into options variable  ##################

#### we can do the df2 here and check if all this works ###

# columns_names= df.columns.tolist()
# options = st.multiselect(
#     'What do you think, which  features are important? Choose your features!',
#     columns_names)

# st.write('You selected:', options)

###  Kidney kids. We have to split the matrix with the subset (options variable)??  ###
#### KIDNEY KIDS, FROM HERE USE THE ORIGINAL MODELS??? ################

# sel = list(options.values())

# df3 = df[sel]

# clf = get_classifier(classifier_name, params)

# clf.fit(X_train_preproc######, y_train)

# y_pred = clf.predict(X_test_preproc######)

# acc = accuracy_score(y_test, y_pred)
# st.write(f"classifier = {classifier_name}")
# st.write(f"Accuracy = {acc}")


### doctor select a new row for a new prediction ####


# FROM HERE WERE TAKEN FROM PREVIOUS WEBPAGE

# age=st.number_input('age',min_value=2, max_value=100, value=22, step=1, format=None, key=None)
# bp=st.number_input('blood pressure (mm/Hg)', min_value=45, max_value=180, value=66, step=1)
# sg=st.number_input('urin specific gravity (sg)',min_value=1.005,max_value=1.025,step=0.005)
# al=st.selectbox('albumin (al): yes (1) no (0)',options=[0,1,2,3,4,5])
# su=st.selectbox('sugar (su): yes (1) no (0)',options=[0,1,2,3,4,5])
# rbc=st.selectbox('red blood care (rbc): abnormal (1) normal (0)',options=[0,1])
# pc=st.selectbox('pus cell (pc): abnormal (1) normal (0)',options=[0,1])
# pcc=st.selectbox('pus cell clumps (pcc): present (1) not present (0)',options=[0,1])
# ba=st.selectbox('bacteria (ba): present (1) not present (0)',options=[0,1])
# bgr=st.number_input('blood gluco random (mgs/dl)',min_value=70, max_value=500, value=131, step=1)
# bu=st.number_input('blood urea (mgs/dl)',min_value=10, max_value=309, value=52, step=1)
# sc=st.number_input('serum creatinine (mgs/dl)',min_value=0.4, max_value=15.2, value=2.2, step=0.1)
# sod=st.number_input('sodium (mEq/L)',min_value=111, max_value=150, value=138, step=1)
# pot=st.number_input('potassium (mEq/L)',min_value=2.5, max_value=47.0, value=4.6, step=0.1)
# hemo=st.number_input('hemoglobin (gms)',min_value=3.1, max_value=17.8, value=13.7, step=0.1)
# pcv=st.number_input('packed cell count (pcv)',min_value=16, max_value=55, value=30, step=1)
# wc=st.number_input('white blood cell count (cells/cumm)',min_value=3000, max_value=15000, value=7000, step=100)
# rc=st.number_input('red blood cell count (millions/cumm)',min_value=2.2, max_value=6.9, value=5.0, step=0.1)
# htn=st.selectbox('hypertension (htn): yes (1) no (0)',options=[0,1])
# dm=st.selectbox('diabetes mellitus (dm): yes (1) no (0)',options=[0,1])
# cad=st.selectbox('coronary artery disease (cad): yes (1) no (0)',options=[0,1])
# appet=st.selectbox('appetite (appet): good (1) poor (0)',options=[0,1])
# pe=st.selectbox('pedal edema (pe): yes (1) no (0)',options=[0,1])
# ane=st.selectbox('anemia (ane): yes (1) no (0)',options=[0,1])

# 10 selected features
age=st.slider('age',min_value=2, max_value=90, value=55)
# bp=st.slider('blood pressure (mm/Hg)', min_value=45, max_value=180)
sg=st.slider('urin specific gravity (sg)',min_value=1.005,max_value=1.025,step=0.005, value=1.02)
# al=st.slider('albumin (al)', min_value=0,max_value=5)
su=st.slider('sugar (su)', min_value=0,max_value=5, value=0)
# rbc=st.selectbox('red blood care (rbc): abnormal (1) normal (0)', options=[0,1])
# pc=st.selectbox('pus cell (pc): abnormal (1) normal (0)', options=[0,1])
# pcc=st.selectbox('pus cell clumps (pcc): present (1) not present (0)', options=[0,1])
ba=st.selectbox('bacteria (ba): present (1) not present (0)', options=[0,1])
bgr=st.slider('blood gluco random (mgs/dl)', min_value=70, max_value=500, value=117)
# bu=st.slider('blood urea (mgs/dl)', min_value=10, max_value=309)
# sc=st.slider('serum creatinine (mgs/dl)', min_value=0.4, max_value=15.2, step=0.1)
# sod=st.slider('sodium (mEq/L)', min_value=111, max_value=150)
# pot=st.slider('potassium (mEq/L)', min_value=2.5, max_value=47.0, step=0.1)
hemo=st.slider('hemoglobin (gms)', min_value=3.1, max_value=17.8, step=0.1, value=13.5)
pcv=st.slider('packed cell count (pcv)', min_value=16, max_value=55, value=41)
wc=st.slider('white blood cell count (cells/cumm)', min_value=3000, max_value=15000, step=100, value=6900)
# rc=st.slider('red blood cell count (millions/cumm)', min_value=2.2, max_value=6.9, step=0.1)
htn=st.selectbox('hypertension (htn): yes (1) no (0)', options=[0,1])
# dm=st.selectbox('diabetes mellitus (dm): yes (1) no (0)', options=[0,1])
# cad=st.selectbox('coronary artery disease (cad): yes (1) no (0)', options=[0,1])
# appet=st.selectbox('appetite (appet): good (1) poor (0)', options=[0,1])
pe=st.selectbox('pedal edema (pe): yes (1) no (0)', options=[0,1])
# ane=st.selectbox('anemia (ane): yes (1) no (0)', options=[0,1])


# make a query:

#### remove brackets ######
# selected_features = {'age':age,'bp':bp,'sg':sg,'al':al,'su':su,'rbc':rbc,'pc':pc,'pcc':pcc,'ba':ba,'bgr':bgr,'bu':bu,'sc':sc,'sod':sod,'pot':pot,'heml':hemo,'pvc':pcv,'wc':wc,'rc':rc,'htn':htn,'dm':dm,'cad':cad,'appet':appet,'pe':pe,'ane':ane}
selected_features = {'age':age,'sg':sg,'su':su,'ba':ba,'bgr':bgr,'hemo':hemo,'pcv':pcv,'wc':wc,'htn':htn,'pe':pe, 'bp': 80, 'bu': 44, 'sc':1.2, 'sod':136, 'pot':4, 'rc':5.2, 'al':0, 'rbc':0, 'pc':0, 'pcc':0, 'dm':0, 'cad': 0, 'appet': 2, 'ane':0}
#selected_features = dict([a, str(x)] for a, x in selected_features.items())

# selected_features = pd.DataFrame({'age':[age],'bp':[bp],'sg':[sg],'al':[al],'su':[su],'rbc':[rbc],'pc':[pc],'pcc':[pcc],'ba':[ba],'bgr':[bgr],'bu':[bu],'sc':[sc],'sod':[sod],'pot':[pot],'heml':[hemo],'pvc':[pcv],'wc':[wc],'rc':[rc],'htn':[htn],'dm':[dm],'cad':[cad],'appet':[appet],'pe':[pe],'ane':[ane]})

print(selected_features)

# selected_features
# result = loaded_model.predict(selected_features)

#### ADD DEFAULT VALUES #######

url = 'https://testimage2-f77cyo2fpq-ew.a.run.app/predict'
#url = 'http://127.0.0.1:8000/predict'

result = requests.get(url, selected_features) ######### UNCOMMENT ########

#outputs prediction and probabilty of prediction from current local api:
#st.write(f"Classification: {result.json()['result']}")
#proba = round(float(result.json()['proba']), 2)
#st.write(f"Prabability: {proba}")


#outputs prediction and probabilty of prediction from current image:
st.write(f" Classification: {result.json()['result']} ")
for i, j in result.json()['proba'].items():
    st.write(f" Probability: {round(j, 5)} ")


# if result==0:
#     answer='You are not at risk of CKD.'
# else:
#     answer='You might be at risk of CKD. Check with your doctor.'

# st.write('')
# check_button = st.button('Check to see if you are at risk of KD?')
# st.subheader('Model predicts:')
# if check_button:
#     st.sidebar.write(answer)
#     st.write(answer)

# Buttons
if st.button('About Us'):
    st.write('Kidney Kids 2022: Viktoria von Laer, Jeanne Mbebi, Markus Kramer, Cristian Jeraldo')
    image3 = Image.open('lewagon.png')
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
