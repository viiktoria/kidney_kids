########### importing libraries ###############
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from PIL import Image
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# for this try are used classifiers and accuracy, would not be necessary if model is taken from our modules

####### importing data and preprocessing #######
from kidney_kids.data import get_cleaned_data
from kidney_kids.randomforest import RandomForest

X_train, X_test, y_train, y_test = get_cleaned_data(path = 'raw_data/kidney_disease.csv')
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
x_values = st.selectbox("X axis", options=X_train_preproc)
y_values = st.selectbox("Y axis", options=X_train_preproc)

plot=px.scatter(data_frame=X_train_preproc, x=x_values, y=y_values) # define df
st.plotly_chart(plot)

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

         **Hyperparameters considered**:\n
         ***C***, float, default=1.0. Inverse of regularization strength; must be a positive float. Like in support vector machines,
         smaller values specify stronger regularization.\n
         ***max_iter***, int, default=100. Maximum number of iterations taken for the solvers to converge.

         """)

#####  User select model and hyperparameters  ##################

classifier_name = st.selectbox("Select Classifier", ("KNN", "Random Forest", "Logistic Regression"))

st.write("Shape of Training Dataset", X_train_preproc.shape)

def add_parameters_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        n_neighbors = st.slider("n_neighbors", 1, 15)
        p = st.selectbox("p", options=[1,2])
        params["n_neighbors"] = n_neighbors
        params["p"] = p
    elif clf_name == "Random Forest":
        max_depth = st.slider("max_depth", 2, 15)
        n_estimators = st.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    else:
        C = st.slider("C", 0.01, 100.0)
        max_iter = st.slider("max_iter", 2, 15)
        params["C"] = C
        params["max_iter"] = max_iter
    return params

#### KIDNEY KIDS, FROM HERE USE THE ORIGINAL MODELS??? ################

params = add_parameters_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["n_neighbors"], p=params["p"])
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(max_depth=params["max_depth"], n_estimators=params["n_estimators"])
    else:
        clf = LogisticRegression(C=params["C"], max_iter=params["max_iter"])

    return clf

clf = get_classifier(classifier_name, params)

clf.fit(X_train_preproc, y_train)

y_pred = clf.predict(X_test_preproc)

acc = accuracy_score(y_test, y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")

#####  Plot actual vs predicted classification  ##################

fig, ax = plt.subplots()
ax.scatter(y_pred, y_test, alpha=0.8, cmap=plt.viridis)
plt.xlabel("CKD predicted")
plt.ylabel("Actual CKD")

st.pyplot(fig)

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

columns_names= df.columns.tolist()
options = st.multiselect(
    'What do you think, which  features are important? Choose your features!',
    columns_names)

st.write('You selected:', options)

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






# Buttons
if st.button('About Us'):
    st.text('Kidney Kids 2022')
