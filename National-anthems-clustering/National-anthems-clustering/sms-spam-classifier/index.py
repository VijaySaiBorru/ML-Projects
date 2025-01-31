import streamlit as st
import string
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear() 
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)       
    text=y[:]
    y.clear() 
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

st.title("Email/SMS Spam Classifier")
st.divider()

input_sms=st.text_input("Enter the message ")


st.divider()

predict = st.button("Predict!")

st.divider()

if predict:
    st.balloons()
    transformed_sms=transform_text(input_sms)
    vector_input=tfidf.transform([transformed_sms])
    result=model.predict(vector_input)[0]
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")

else:
    "Please use the button for prediction" 