# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import streamlit as st 
from joblib import dump, load

def generate_data(x, models):
    
    '''This function generates metadata with k predictions of k base learners for custom model'''
    
    res_x = []
    for model in models:
        res_x.append(model.predict(x))
    res_x = np.array(res_x).T

    return res_x

def predict_fraud(X):
    
    '''This function takes details about a healthcare provider as input and returns a prediction of the healthcare provider
       being a potential fraud. The details include: no. of inpatient claims(is_inpatient), no. of claims with group codes
       (is_groupcode), no. of claims with chronic illnesses like heartfailure, alzeimer, diabetes, etc., avg. deductible amt,
       avg. insurance amount reimbursed to the provider and avg. no. of days a patient was admitted under provider's care.'''
    
    # Loading Standard Scaler model to scale the data
    path = os.getcwd()
    with open (path+'\\Saved_Models\\scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    # Deleting provider id
    X = np.delete(X, 0, 1)
    
    # Scaling data
    X_scaled = scaler.transform(X)
    
    # Loading all base learners
    files = os.listdir(path+'\\Saved_Models\\base_learners2')
    models = []
    for model in files:
        clf = load(path+'\\Saved_Models\\base_learners2\\'+model)
        models.append(clf)
        
    # Loading custom model
    custom_model = load(path+'\\Saved_Models\\best_custom_model2.joblib')
    
    # Predictions
    x_meta = generate_data(X_scaled, models)
    y_pred = custom_model.predict(x_meta)
    y_prob = custom_model.predict_proba(x_meta)
    
    return y_pred, y_prob

def main():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Healthcare Provider Fraud Predictor </h2>
    </div>
    <br>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.write("Please enter the following details of provider to know if he/she is fraud.")
    st.write("")
    provider_id = st.text_input("Provider Id","")
    inpatient_claims = st.text_input("No. of  inpatient claims","")
    groupcode_claims = st.text_input("No. of claims with group codes","")
    chronic_rheuma = st.text_input("No. of claims with rheumatoidarthritis","")
    benefic_count = st.text_input("No. of beneficiaries of provider","")
    deductible_amt = st.text_input("Average deductible amount", "")
    amount_reimursed = st.text_input("Average claim amount reimbursed", "")
    chronic_alzheimer = st.text_input("No. of claims with alzheimer","")
    chronic_chemicheart = st.text_input("No. of claims with chemical heart","")
    days_admitted = st.text_input("Average no. of days a patient was admitted under provider's care","")
    chronic_stroke = st.text_input("No. of claims with stroke","")
    
    input_array = np.array([provider_id, inpatient_claims, groupcode_claims,
                            chronic_rheuma, benefic_count, deductible_amt,
                            amount_reimursed, chronic_alzheimer, chronic_chemicheart,
                            days_admitted, chronic_stroke]).reshape(1, -1)
    
    res, prob = "", ""
    if st.button("Predict"):
        y_pred, y_prob =predict_fraud(input_array)
        if y_pred == 1:
            res = 'Fraud'
            prob = y_prob[:, 1][0]*100
        else:
            res = 'Not Fraud'
            prob =y_prob[:, 0][0]*100
            
        st.write('Health Care Provider with ID {} is {} with probability {}%'\
               .format(provider_id, res, prob))

if __name__=='__main__':
    main()
    
    


