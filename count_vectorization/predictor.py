
import streamlit as st
import joblib

st.title("Real-Fake classifier : ")
text_model=joblib.load('C:/Users/jnave/Dropbox/My PC (LAPTOP-USDFL75P)/Downloads/Real-Fake')
inp=st.text_input("Enter the message : ")
opt=text_model.predict([inp])
if st.button('Predict'):
  st.title("The message entered is : "+opt[0])
