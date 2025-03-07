import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the AI4I dataset
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

# Evaluate model performance
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# 1️⃣ Visualizing Feature Importance (Regression Coefficients)
plt.figure(figsize=(10, 5))
coefficients = pd.Series(model.coef_, index=X.columns)
coefficients.plot(kind='bar', color='blue')
plt.title("Feature Importance (Regression Coefficients)")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.grid()
plt.show()

# 2️⃣ Residual Plot (Errors should be normally distributed)
residuals = Y_test - Y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True, color="red")
plt.title("Residual Distribution (Errors)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# 3️⃣ Actual vs. Predicted Plot
plt.figure(figsize=(8, 5))
plt.scatter(Y_test, Y_pred, alpha=0.5, color='green')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r', linestyle="--")  # Perfect fit line
plt.xlabel("Actual Tool Wear")
plt.ylabel("Predicted Tool Wear")
plt.title("Actual vs Predicted Tool Wear")
plt.grid()
plt.show()
