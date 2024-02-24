import pandas as pd
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
unemployment_data = pd.read_csv('/content/drive/MyDrive/Unemployment in India.csv')

# Check the first few rows of the dataframe
print(unemployment_data.head())

# Data preprocessing
# You might need to clean data, convert data types, or create additional features here
print(unemployment_data.columns)
print(unemployment_data.dtypes)
# Convert ' Date' column to datetime format
unemployment_data[' Date'] = pd.to_datetime(unemployment_data[' Date'])

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
plt.plot(unemployment_data[' Date'], unemployment_data[' Estimated Unemployment Rate (%)'])

plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.tight_layout()
plt.show()



# Regional Analysis
import seaborn as sns

# Calculate average unemployment rate for each region
avg_unemployment_by_region = unemployment_data.groupby('Region')[' Estimated Unemployment Rate (%)'].mean().reset_index()

# Visualize average unemployment rate by region
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_unemployment_by_region, x=' Estimated Unemployment Rate (%)', y='Region')
plt.title('Average Unemployment Rate by Region')
plt.xlabel('Average Unemployment Rate (%)')
plt.ylabel('Region')
plt.show()





# Correlation Analysis
correlation_matrix = unemployment_data.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()





# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(unemployment_data[' Estimated Unemployment Rate (%)'], bins=20, kde=False, color='skyblue')
plt.title('Histogram of Estimated Unemployment Rate')
plt.xlabel('Estimated Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.show()






# Density Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(unemployment_data[' Estimated Unemployment Rate (%)'], shade=True, color='orange')
plt.title('Density Plot of Estimated Unemployment Rate')
plt.xlabel('Estimated Unemployment Rate (%)')
plt.ylabel('Density')
plt.show()





# Counter Plot (Count Plot)
plt.figure(figsize=(10, 6))
sns.countplot(data=unemployment_data, x='Region', palette='Set2')
plt.title('Count of Data Points by Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()
