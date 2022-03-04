import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from kidney_kids.data import get_cleaned_data
from kidney_kids.randomforest import RandomForest

X_train, X_test, y_train, y_test = get_cleaned_data(path = 'raw_data/kidney_disease.csv')
forest_model = RandomForest()
X_preproc = forest_model.preproc(X_train)


# Title/Text
# st.title('Chronic Kidney Disease')
# st.text('Built with Streamlit')

# # Header and Subheader
# st.header('EDA Section')
# st.subheader('kidney_disease.csv')

# # Checkbox
# if st.checkbox("Show Dataset"):
#     st.text('Showing Dataset')

# # Selection
# appetite = st.selectbox('Appetite', ('Good','Poor'))

# # Sliders
# age = st.slider('Your Age',2,90)

# # Buttons
# if st.button('About Us'):
#     st.text('Kidney Kids 2022')

if __name__ == '__main__':
    print(X_preproc)
