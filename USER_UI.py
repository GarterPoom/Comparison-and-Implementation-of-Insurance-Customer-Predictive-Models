import streamlit as st
import pandas as pd
import pickle

# Load the data
df = pd.read_csv("insurance.csv")

# Preprocess the data
X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["charges"]
X = pd.get_dummies(X)  # One-hot encode categorical variables

# Create a form for the user to input the necessary information
age = st.number_input("What is your age?", min_value=0, max_value=120)
sex = st.radio('Gender :', ['male', 'female'])
smoker = st.radio("Are you a smoker?", ['No', 'Yes'])
region = st.selectbox("Select your region", ["northwest", "northeast", "southeast", "southwest"])
children = st.number_input("How many children do you have?", min_value=0, max_value=5)
bmi = st.number_input("What is your BMI?", min_value=0, max_value=50)

# Predict the charges based on the input information
inputs = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "children": [children],
    "bmi": [bmi],
    "smoker": [smoker],
    "region": [region]
})
inputs = pd.get_dummies(inputs)  # One-hot encode categorical variables

#Load normalization
load_nor = pickle.load(open('normalization.pkl', 'rb'))

#Apply the normalization model to new data
inputs = load_nor.transform(inputs)

#Deploy Random Forest Regression
load_RAN = pickle.load(open('regression.pkl', 'rb'))

#Apply the Random Forest Regression to new data
prediction = load_RAN.predict(inputs)[0]

# Display the prediction
st.write(f"Predicted charges: ${prediction:.2f}")
