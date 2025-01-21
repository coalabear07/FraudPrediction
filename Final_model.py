# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:19:15 2025

@author: hp
"""
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import gdown
import os

# Define the URL for the Google Drive file
file_id = '1Sbgz9Z2T63ufQn2z9f60GWyv0Q2MpKWk'  # Replace with your actual file ID
gdown_url = f'https://drive.google.com/uc?id={file_id}'

# Specify the path where you want to save the model file
model_path = './RandomForest.pkl'

# Download the model file if it doesn't already exist
if not os.path.exists(model_path):
    gdown.download(gdown_url, model_path, quiet=False)

# Load the model
with open(model_path, 'rb') as model_file:
    fraud_model = pickle.load(model_file)

scaler = pickle.load(open("scaler.sav",'rb'))



# sidebar for navigate

with st.sidebar:
    
    selected = option_menu("Table",
                          ["Introduction" ,
                           "Fraud Transaction Prediction"], 
                          icons = ["journal","credit-card"],
                          default_index = 0
                          )
#If Introduction Page
if (selected == "Introduction") :
    
    st.title("Introduction to Fraud Transaction Prediction")
    
    # Add a header
    st.header("Overview")
    
    st.write("""
    This application is built using the **Credit Card Transactions Fraud Detection Dataset** (taken from Kaggle Platform), a comprehensive dataset designed for fraud detection in financial transactions. 
    The dataset is divided into two parts:
    - **Training Set**: Contains 1.3 million transaction records used to train the predictive model.
    - **Test Set**: Comprises 550,000 transaction records for evaluating the model's performance.
    The data includes features that capture the transaction details and associated behaviors, enabling the detection of potential fraudulent activities.
    """)
    
    # Add an image related to the dataset
    st.image("Dataset.png", caption="Credit Card Fraudulent Dataset", use_container_width=True)

    # Information Paragraph 2: About the Model
    st.write("""
    To ensure high accuracy and reliability in detecting fraudulent transactions, this application uses a **Random Forest Classifier** as the core predictive model. 
    The Random Forest algorithm was chosen due to its superior performance in handling large datasets, generating robust predictions, and achieving high overall evaluation metrics such as precision, recall, and F1-score.
    The following diagrams shows the results of all the models tested using test dataset where Random Forest Models performed the best out of all the models.
    """)
    # Add an image related to the dataset
    st.image("Models Performance.png", caption="Model Performance on the Dataset", use_container_width=True)
    
    st.write("""
    **Random Forest** :
    - Accuracy : 99.97%
    - Precision : 64.23%
    - Recall : 61.54%
    - F1-score : 62.86%
    - MCC : 62.73%
    - ROC-AUC : 96.36%	
    - PR-AUC : 65.15%
    
    """)
	
    
    # Information Paragraph 3: Information about features
    st.write("""
    In this predictive model, several features used in order to predict or classify fraud transaction.The following is the features list used in deployment phases :
    - **Month of Transaction**
    - **Year of Transaction**
    - **Age of Card Holder**
    - **Amount of Transaction**
    - **Category of Transaction for**
    """)
    
    

#If Fraud Prediction Page
if (selected == "Fraud Transaction Prediction") :
    
    st.title("Fraudulent Transaction Prediction")
    
    # Input order : amt,age,trans_month,trans_year,category_food_dining,category_gas_transport
    #category_grocery_net,category_grocery_pos,category_health_fitness,category_home,category_kids_pets
    #category_misc_net,category_misc_pos,category_personal_care,category_shopping_net,
    #category_shopping_pos,category_travel
    
         
    #users input values 
    st.write(""" Disclaimers : The predictive models are not 100% accurate """)
    Month = st.text_input("Month of Transaction", placeholder="e.g. 08 ")
    Year = st.text_input("Year of Transaction",placeholder="e.g. 2023 ")
    Age = st.text_input("Age of Card Holder",placeholder="e.g. 35 ")
    Amount = st.text_input("Amount of Transaction",placeholder="e.g. 700 ")
    

    # Dropdown for category selection
    categories = [
        "Food Dining", "Gas Transport", "Grocery Net",
        "Grocery Pos", "Health Fitness", "Home",
        "Kids Pets", "Misc Net", "Misc Pos",
        "Personal Care", "Shopping Net", "Shopping Pos",
        "Travel","Others"
    ]
    selected_category = st.selectbox("Select the transaction category:", categories)

    if(selected_category == "Food Dining"):
        input_data = [Amount,Age,Month,Year,1,0,0,0,0,0,0,0,0,0,0,0,0]
    elif(selected_category == "Gas Transport"):
        input_data = [Amount,Age,Month,Year,0,1,0,0,0,0,0,0,0,0,0,0,0]
    elif(selected_category == "Grocery Net"):
        input_data = [Amount,Age,Month,Year,0,0,1,0,0,0,0,0,0,0,0,0,0]
    elif(selected_category == "Grocery Pos"):
        input_data = [Amount,Age,Month,Year,0,0,0,1,0,0,0,0,0,0,0,0,0]
    elif(selected_category == "Health Fitness"):
        input_data = [Amount,Age,Month,Year,0,0,0,0,1,0,0,0,0,0,0,0,0]
    elif(selected_category == "Home"):
        input_data = [Amount,Age,Month,Year,0,0,0,0,0,1,0,0,0,0,0,0,0]
    elif(selected_category == "Kids Pets"):
        input_data = [Amount,Age,Month,Year,0,0,0,0,0,0,1,0,0,0,0,0,0]
    elif(selected_category == "Misc Net"):
        input_data = [Amount,Age,Month,Year,0,0,0,0,0,0,0,1,0,0,0,0,0]
    elif(selected_category == "Misc Pos"):
        input_data = [Amount,Age,Month,Year,0,0,0,0,0,0,0,0,1,0,0,0,0]
    elif(selected_category == "Personal Care"):
        input_data = [Amount,Age,Month,Year,0,0,0,0,0,0,0,0,0,1,0,0,0]
    elif(selected_category == "Shopping Net"):
        input_data = [Amount,Age,Month,Year,0,0,0,0,0,0,0,0,0,0,1,0,0]
    elif(selected_category == "Shopping Pos"):
        input_data = [Amount,Age,Month,Year,0,0,0,0,0,0,0,0,0,0,0,1,0]
    elif(selected_category == "Travel"):
        input_data = [Amount,Age,Month,Year,0,0,0,0,0,0,0,0,0,0,0,0,1]
    else:
        input_data = [Amount,Age,Month,Year,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
        
    diagnosis = ''
    
    # Prediction button
    if st.button("Transaction Test Result"):
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        input_data_scaled = scaler.transform(input_data_reshaped)
        diagnosis = fraud_model.predict(input_data_scaled)
   
        if (diagnosis[0] == 0):
            diagnosis = "This Transaction is potentially Legit Transaction"
        else:
            diagnosis = "This Transaction is potentially Fraudulent Transaction"
        
    st.success(diagnosis)
    
  
    
