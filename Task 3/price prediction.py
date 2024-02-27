from google.colab import drive
drive.mount('/content/drive')
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
car_data = pd.read_csv('/car_price_prediction.csv')  # Assuming 'car_data.csv' contains your dataset

# Data preprocessing
print(car_data.columns)
# Assuming 'car_data.csv' contains columns like 'year', 'mileage', 'horsepower', 'price'
 # Features
 # Remove non-numeric characters and convert to float
car_data['Mileage'] = car_data['Mileage'].str.replace(' km', '').astype(float)
# Remove non-numeric characters and convert to float
car_data['Engine volume'] = car_data['Engine volume'].str.replace(' Turbo', '').astype(float)

X = car_data[['Prod. year', 'Mileage', 'Engine volume']]
y = car_data['Price']  # Target variable


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Predicting car prices
# Assuming you have a new car's features in a DataFrame called 'new_car_features'
# Assuming you have a new car's features in a DataFrame called 'new_car_features'
new_car_features = pd.DataFrame([[2022, 50000, 200]], columns=['Prod. year', 'Mileage', 'Engine volume'])
predicted_price = model.predict(new_car_features)
print(f'Predicted price for the new car: ${predicted_price[0]}')
