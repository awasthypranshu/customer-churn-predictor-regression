import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import pickle
import datetime
import streamlit as st
with open("gender_encoder.pkl","rb") as file:
    gender_encoder = pickle.load(file)

with open("geo_ohe.pkl","rb") as file:
    geo_ohe = pickle.load(file)

with open("scaler.pkl","rb") as file:
    scaler = pickle.load(file)

model = tf.keras.models.load_model("RegressionModel.h5")

# Streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox(
    'Geography',
    geo_ohe.categories_[0]
)
gender = st.selectbox(
    'Gender',
    gender_encoder.classes_
)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = {
    'CreditScore':[credit_score],
    'Geography': [geography],
    'Gender': [gender_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}


data = pd.DataFrame(input_data)
encoded_geography = geo_ohe.transform(data[["Geography"]]).toarray()
geo_df = pd.DataFrame(encoded_geography,columns=geo_ohe.get_feature_names_out())

data = pd.concat([data.drop("Geography",axis=1),geo_df],axis=1)

scaled_data = scaler.transform(data)
prediction = model.predict(scaled_data)

st.write(f"The probability of customer churning is {prediction[0][0]:.2f}")