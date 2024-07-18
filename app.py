import streamlit as st
import pandas as pd
import numpy as np
from src.data_processing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv('data/CarPrice_Assignmnet.csv')

# Preprocess data
df = preprocess_data(df)
df = df.fillna(df.mean())

df = pd.get_dummies(df, drop_first=True)

return df

# Split data
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Streamlit UI
st.title("Car Price Prediction")

st.write("### Mean Absolute Error", mae)
st.write("### Mean Squared Error", mse)
st.write("### Root Mean Squared Error", rmse)

st.write("### Sample Predictions")
st.write(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head())
