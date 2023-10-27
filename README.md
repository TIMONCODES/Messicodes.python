# Messicodes.python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load your dataset
data = pd.read_csv("accident_data.csv")

# Define the dependent and independent variables
X = data[['Weather', 'Road Type', 'Speed Limit', 'Vehicle Type']]
y = data['Accident Severity']
  
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model for future use
from joblib import dump
dump(model, 'accident_severity_model.joblib')
