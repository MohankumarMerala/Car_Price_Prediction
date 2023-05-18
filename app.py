#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pickle import dump, load

# For Suppressing warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import ExtraTreesRegressor

# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# In[2]:


import streamlit as st




# In[6]:


st.title('Model Deployment: Car Price Prediction')


Present_Price = st.number_input("Enter Present_Price")

Kms_Driven = st.number_input("Enter Kms_Driven")

Owner = st.number_input("Enter Number of Existing Owner")

no_year = st.number_input("Enter Number of years Car used")

Fuel_Type_Diesel = st.number_input("Enter Fuel_Type_Diesel(Type 1 if present or 0 )",min_value=0,max_value=1)

Fuel_Type_Petrol = st.number_input("Enter Fuel_Type_Petrol(Type 1 if present or 0 )",min_value=0,max_value=1)

Seller_Type_Individual = st.number_input("Enter Seller_Type_Individual(if dealer(Type 0) or individual(Type 1))",min_value=0,max_value=1)

Transmission_Manual = st.number_input("Transmission_Manual(if Manual(Type 0) or Automatic(Type 1)",min_value=0,max_value=1)




loaded_model = load(open("final_model.sav", 'rb'))

list1 = [Present_Price,Kms_Driven,Owner,no_year,Fuel_Type_Diesel,Fuel_Type_Petrol,Seller_Type_Individual,Transmission_Manual]
import numpy as np

# reshape the input data to a 2D array
list1 = np.array(list1).reshape(1, -1)

# predict the price  for the input data
result = loaded_model.predict(list1)

# display the predicted value
submit = st.button('Submit')
if submit:
    if result<0:
        st.write("Don't Purchase this car")

    else:
        st.write("The price of the car is{}".format(result))

