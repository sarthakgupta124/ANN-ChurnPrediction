import streamlit as st
import pandas as pd 
import numpy as np
import pickle
from tensorflow.keras.models import load_model
st.title("Customer Churn Prediction")
st.write("Enter the customer details to predict churn probability")
# Define input fields

CreditScore=st.number_input("Credit Score")
Geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
Age=st.slider("Age",18,92,40)
Tenure=st.slider("Tenure",0,10,3)
balance=st.number_input("Balance")
NumOfProducts=st.slider("Number of Products",1,4,2)
HasCrCard=st.selectbox("Has Credit Card", [0, 1])
IsActiveMember=st.selectbox("Is Active Member", [0, 1]) 
EstimatedSalary=st.number_input("Estimated Salary")

input_data={
    'CreditScore': CreditScore,
    'Geography': Geography,
    'Gender': gender,   
    'Age': Age,
    'Tenure': Tenure,
    'Balance': balance,
    'NumOfProducts': NumOfProducts,
    'HasCrCard': HasCrCard,
    'IsActiveMember': IsActiveMember,

    'EstimatedSalary': EstimatedSalary

}
if st.button("Predict Churn Probability"):
    ## Load the saved model,label encoder and preprocessor
    model = load_model('model.h5')
    with open('label_encoder_gender.pkl','rb') as f:
        label   = pickle.load(f)        
    with open('preprocessor.pkl','rb') as f:    
        preprocessor = pickle.load(f)
    
    input_df = pd.DataFrame([input_data])
    # Preprocess the input data     
    input_df['Gender'] = label.transform(input_df['Gender'])
    input_processed = preprocessor.transform(input_df)  
    #make prediction
    prediction = model.predict(input_processed)
    prediction_proba = prediction[0][0]
    if prediction_proba > 0.5:
        st.write(f"The customer is likely to churn with a probability of {prediction_proba:.2f}")   

    else:
        st.write(f"The customer is unlikely to churn with a probability of {1 - prediction_proba:.2f}")

