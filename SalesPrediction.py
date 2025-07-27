import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Loading the dataset
data = pd.read_csv('D:\Internship projects/Advertising.csv')

# Exploring the data
print("First 5 rows:")
print(data.head())
print("\nData Summary:")
print(data.describe())
print("\nMissing Values:")
print(data.isnull().sum())

# Visualizing relationships
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='reg')
plt.suptitle('Feature vs Sales')
plt.tight_layout()
plt.show()

# Preparing the data
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Training-testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_}")
print(f"TV: {model.coef_[0]}")
print(f"Radio: {model.coef_[1]}")
print(f"Newspaper: {model.coef_[2]}")

print("\nModel Performance:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Predicting new data
new_data = pd.DataFrame({'TV': [100], 'Radio': [25], 'Newspaper': [15]})
predicted_sales = model.predict(new_data)
print(f"\nPredicted Sales for new data: {predicted_sales[0]:.2f}")
