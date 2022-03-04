import streamlit as st

# Title/Text
st.title('Chronic Kidney Disease')
st.text('Built with Streamlit')

# Header and Subheader
st.header('EDA Section')
st.subheader('kidney_disease.csv')

# Checkbox
if st.checkbox("Show Dataset"):
    st.text('Showing Dataset')

# Selection
appetite = st.selectbox('Appetite', ('Good','Poor'))

# Sliders
age = st.slider('Your Age',2,90)

# Buttons
if st.button('About Us'):
    st.text('Kidney Kids 2022')
