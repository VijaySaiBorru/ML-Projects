import streamlit as st
import numpy as np
import joblib

sc=joblib.load("Scaler.pkl")
model=joblib.load("model.pkl")

st.title("Real Estate Price Prediction")
st.divider()

bed=st.number_input("Enter the number of bedrooms",value=2,step=1)
bath=st.number_input("Enter the number of bathrooms",value=1,step=1)
size=st.number_input("Enter the size",value=1000,step=50)

X=[bed,bath,size]
st.divider()

predict = st.button("Predict!")

st.divider()

if predict:
    st.balloons()
    X1=np.array(X)
    X_array = sc.transform([X1])
    y_pred=model.predict(X_array)[0]
    st.write(f"The prediction is {y_pred: .2f}")


else:
    "Please use the button for prediction"    
#['bed', 'bath', 'house_size']