


import streamlit as st
import numpy as np
import pickle
import time

# Set the page title and favicon
st.set_page_config(page_title="Revenue Radar", page_icon="📊")

# Title of the app
st.title("📊 **Revenue Radar**")
st.subheader("🔮 Predict Yearly Customer Spending")

# Add some explanation and styling
st.markdown("""
Welcome to **Revenue Radar**! 🧭
This tool helps predict the **yearly amount spent by a customer** based on their app usage and membership details.
""")
st.markdown("### Enter Customer Data to make a prediction:")

# Use the sidebar for inputs
with st.sidebar:
    st.header("Customer Data Inputs")
    avg_session_length = st.number_input("💻 Average Session Length (minutes)", min_value=0.0, step=0.1, help="How long the customer spends per session.")
    time_on_app = st.number_input("📱 Time on App (minutes)", min_value=0.0, step=0.1, help="Total time spent on the app.")
    length_of_membership = st.number_input("🗓 Length of Membership (years)", min_value=0.0, step=0.1, help="How long the customer has been a member.")
    st.markdown("---")

# Load the scaler and model using pickle
try:
    with open('models/scaler.pkl', 'rb') as scaler_file:
        loaded_scaler = pickle.load(scaler_file)
    with open('models/model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
except FileNotFoundError:
    st.error("❌ Error: Model files not found! Ensure 'scaler.pkl' and 'model.pkl' exist in the 'models' directory.")
    st.stop()

# Prediction button with loading spinner
if st.button("🔍 Predict"):
    with st.spinner('Predicting... Please wait!'):
        time.sleep(2)  # Simulate some time delay for the prediction
        try:
            # Prepare input data for prediction
            input_data = np.array([avg_session_length, time_on_app, length_of_membership]).reshape(1, -1)
            
            # Scale the input data
            scaled_data = loaded_scaler.transform(input_data)
            
            # Make the prediction
            prediction = loaded_model.predict(scaled_data)
            
            # Display the result with added visuals
            st.success(f"💵 Predicted Yearly Amount Spent: **${prediction[0]:,.2f}**")
            st.balloons()  # Add a celebratory balloon effect when prediction is made
        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {str(e)}")

# Add a footer for additional info
st.markdown("""
---
Created by [Your Name] | Powered by Streamlit 🚀
""")




