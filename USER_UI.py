import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Insurance Visualization", page_icon="📊")

st.write('''# Insurance Cost Predictive Webpage''')

# Load the data
df = pd.read_csv("insurance.csv")

# Visualization
st.subheader('Trend of Charge by Age')
dfAGE = pd.DataFrame(df[20:70], columns=['charges', 'age'])
st.line_chart(dfAGE,x='age',y='charges', height=210)
st.info('Follow by age, charge has main relation with age in increasing when age of customers has been increase. That mean if you have older age, you will be need more than charge to paid.')

st.subheader('Trend of Charge by BMI')
dfBMI = pd.DataFrame(df[20:70], columns=['charges', 'bmi'])
st.line_chart(dfBMI, x='bmi',y='charges', height=210)
st.info('BMI has trend similar with age as well as when compare with charge. If you have much BMI, you will get more charge of cost because who have higher BMI your health more risk than well.')

# Select the columns to include in the chart
st.subheader('Sex count')
sex_count = pd.DataFrame(df['sex'].value_counts()).reset_index()
sex_count.columns = ['sex', 'count']

fig = px.bar(
        sex_count,
        x="sex",
        y="count",
        title="Books Read by Year",
        color_discrete_sequence=["#9EE6CF"],
    )
st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# Preprocess the data
X = df[["age", "sex", "children", "bmi", "smoker", "region"]]
y = df["charges"]
df = pd.get_dummies(df)  # One-hot encode categorical variables
X = pd.get_dummies(X)# One-hot encode categorical variables
mms = MinMaxScaler()
df = mms.fit_transform(df)
X = mms.fit_transform(X)

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
#mse = mean_squared_error(y_test, y_pred)

st.sidebar.write('''# Please insert your information''')

def user_input_data():
    # Create a form for the user to input the necessary information
    age = st.sidebar.number_input("What is your age?", min_value=0, max_value=120, value=20)
    children = st.sidebar.slider("How many children do you have?", min_value=0, max_value=5)
    bmi = st.sidebar.number_input("What is your BMI?", min_value=0.0, max_value=50.0, value=25.0)
    sex = st.sidebar.radio("Select your sex", ["female", "male"])
    smoker = st.sidebar.radio("Are you a smoker?",["no", "yes"])
    region = st.sidebar.selectbox("Select your region", ["southeast", "southwest", "northwest", "northeast"])

    # Predict the charges based on the input information
    inputs = {
    "age": age,
    "children": children,
    "bmi": bmi,
    "sex": sex,
    "smoker": smoker,
    "region": region,
    }
    features = pd.DataFrame(inputs, index=[0])
    return features

df = user_input_data()
data_sample = pd.read_csv('insurance_sample.csv')
df2 = pd.concat([df, data_sample], axis=0)

#one hot encoding
cat_data = pd.get_dummies(df2[['sex', 'smoker', 'region']])

#Combine with data input
X_new = pd.concat([cat_data, df2], axis=1)
X_new = X_new[:1]

#Drop un used feature 
X_new = X_new.drop(columns=['sex', 'smoker', 'region', 'smoker'])

inputs = X_new  # One-hot encode categorical variables
inputs = mms.transform(inputs)  # Scale the inputs
#st.write(inputs)
prediction = model.fit(X,y).predict(inputs)[0]

# Display the prediction
st.subheader('Prediction Cost (US Dollar $)')
st.write(prediction)
st.write('-----')
