# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 18:04:31 2025

@author: hp
"""


import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

scaler = pickle.load(open("C:/Users/hp/JPN/scaler.sav",'rb'))

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
    
#If Fraud Prediction Page
if (selected == "Fraud Transaction Prediction") :
    
    st.title("Fraudulent Transaction Prediction")