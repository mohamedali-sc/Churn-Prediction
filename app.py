import streamlit as st
import pickle
import numpy as np


def load_model():
    with open("savedmodel.pkl","rb") as file:
        data=pickle.load(file)
    return data
data=load_model()
model=data["model"]
le=data["le"]
le2=data["le2"]
# X=np.array([[619,"France","Female",42,2,0,1,1,1,101348.88]])
# X[:, 1] = le2.transform(X[:, 1])
# X[:, 2] = le.transform(X[:, 2])
# X[:, -1] = X[:, -1].astype(float)
# X[:, 5] = X[:, 5].astype(float)
# result=model.predict(X)[0]
# print(result)

def predict_show_page():
    st.title("Churn Prediction")
    Geography=("France","Germany","Spain")
    gender=("Female","Male")
    CreditScore=st.text_input("CreditScore")
    Geography=st.selectbox("Geography",Geography)
    Gender = st.selectbox("Gender", gender)
    age=st.slider("Age",0,70,10)

    Tenure = st.text_input("Tenure")
    Balance = st.text_input("Balance")
    NumOfProducts = st.slider("NumOfProducts",0,10,1)
    HasCrCard=st.text_input("HasCrCard")
    IsActiveMember=st.text_input("IsActiveMember")
    EstimatedSalary=st.text_input("EstimatedSalary")
    ok=st.button("Status")
    if ok:
        X = np.array([[CreditScore, Geography, Gender, age, Tenure, Balance,NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]])
        X[:, 1] = le2.transform(X[:, 1])
        X[:, 2] = le.transform(X[:, 2])
        X[:, -1] = X[:, -1].astype(float)
        X[:, 5] = X[:, 5].astype(float)
        result=model.predict(X)[0]
        if result==1:
            st.subheader("The customer will stayed")
        else:
            st.subheader("The customer will exited")



predict_show_page()
