
import streamlit as st
import joblib
import pandas as pd
import re
import string

st.title('Real and Fake news Classifier')
model1=joblib.load('model1_LR')
model2=joblib.load('model2_DT')
model3=joblib.load('model3_RFC')
vectorization=joblib.load('vectorization')

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

def output_lable(n):
    return True if n else False
    
def test(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = model1.predict(new_xv_test)
    pred_DT = model2.predict(new_xv_test)
    pred_RFC = model3.predict(new_xv_test)

    return output_lable(pred_LR[0]),output_lable(pred_DT[0]),output_lable(pred_RFC[0])


inp=st.text_input("Enter the message : ")
opt1,opt2,opt3=test(inp)
if (opt1 and opt2) or (opt2 and opt3) or (opt3 and opt1) or (opt1 and opt2 and opt3):
    ans='True News'
else:
    ans='False News'
if st.button('Predict'):
    st.write(inp)
    st.title("The message entered is : "+ans)
