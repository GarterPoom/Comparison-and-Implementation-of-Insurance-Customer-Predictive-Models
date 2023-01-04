import streamlit as st
import pandas as pd
#from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv("insurance.csv")

# Preprocess the data
X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["charges"]
X = pd.get_dummies(X)  # One-hot encode categorical variables
#std = StandardScaler()
#X = std.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cluster the data using DBSCAN
dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_train)

# Train a random forest regression model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Test the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

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
#inputs = std.transform(inputs)  # Scale the inputs
prediction = model.predict(inputs)[0]

# Display the prediction
st.write(f"Predicted charges: ${prediction:.2f}")
