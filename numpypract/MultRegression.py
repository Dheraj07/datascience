# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("ai4i2020.csv")

# Drop unnecessary columns (Machine ID is not useful for regression)
df = df.drop(columns=['UDI', 'Product ID'])

# Convert categorical 'Type' column into numeric using one-hot encoding
df = pd.get_dummies(df, columns=['Type'], drop_first=True)

# Define independent variables (X) and target variable (Y)
X = df.drop(columns=['Tool wear [min]'])
Y = df['Tool wear [min]']

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train a Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Make predictions
Y_pred = model.predict(X_test)

# Compute R-squared
r2 = r2_score(Y_test, Y_pred)

# Compute Adjusted R-squared
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Print evaluation metrics
print(f"R² Score: {r2:.4f}")
print(f"Adjusted R² Score: {adj_r2:.4f}")
print(f"Mean Squared Error: {mean_squared_error(Y_test, Y_pred):.4f}")
