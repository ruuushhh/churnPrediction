#Import libraries
import streamlit as st
import pandas as pd

#load the model from disk
import joblib
model = joblib.load(r"model.sav")

#Import python scripts
from preprocessing import preprocess

def main():
    #Setting Application title
    st.title('Customer Churn Prediction App')

    st.info("Input data below")
    #Based on our optimal features selection
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Age = st.slider('Age of the Customer', min_value=20, max_value=100, value=20)
    Subscription_Length_Months = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)
    Monthly_Bill = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150, value=0)
    Total_Usage_GB = st.number_input('The total amount charged to the customer',min_value=0, max_value=10000, value=0)

    data = {
            'Age': Age,
            'Gender': Gender,
            'Subscription_Length_Months': Subscription_Length_Months,
            'Monthly_Bill': Monthly_Bill,
            'Total_Usage_GB': Total_Usage_GB
            }
    features_df = pd.DataFrame.from_dict([data])
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.write('Overview of input is shown below')
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.dataframe(features_df)
    #Preprocess inputs
    preprocess_df = preprocess(features_df, 'Online')

    prediction = model.predict(preprocess_df)

    if st.button('Predict'):
        if prediction == 1:
            st.warning('Yes, the customer will terminate the service.')
        else:
            st.success('No, the customer is happy with Telco Services.')


if __name__ == '__main__':
        main()