import pandas as pd

# Load the datasets
items_df = pd.read_csv('/content/item.csv')
sales_df = pd.read_csv('/content/sales.csv')
promotion_df = pd.read_csv('/content/promotion.csv')
supermarkets_df = pd.read_csv('/content/supermarkets.csv')

# Display basic information and the first few rows to understand each dataset
print("Items DataFrame:")
print(items_df.head())
print(items_df.info())

print("\nSales DataFrame:")
print(sales_df.head())
print(sales_df.info())

print("\nPromotion DataFrame:")
print(promotion_df.head())
print(promotion_df.info())

print("\nSupermarkets DataFrame:")
print(supermarkets_df.head())
print(supermarkets_df.info())

# Handling missing values and duplicates
items_df.dropna(inplace=True)  # Drop rows with any missing values in items_df
sales_df.fillna({'CustomerId': 'Unknown'}, inplace=True)  # Fill missing CustomerId with 'Unknown'
promotion_df.drop_duplicates(inplace=True)  # Remove duplicate rows in promotion_df

import pandas as pd
# Load the datasets
items_df = pd.read_csv('/content/item.csv')
sales_df = pd.read_csv('/content/sales.csv')
promotion_df = pd.read_csv('/content/promotion.csv')
supermarkets_df = pd.read_csv('/content/supermarkets.csv')

# Print column names to check for the 'Code' column
print("Items DataFrame columns:", items_df.columns)
print("Sales DataFrame columns:", sales_df.columns)
print("Promotion DataFrame columns:", promotion_df.columns)
print("Supermarkets DataFrame columns:", supermarkets_df.columns)

# Renaming columns to ensure consistency
items_df.rename(columns={'code': 'Code'}, inplace=True)
sales_df.rename(columns={'code': 'Code', 'supermarket': 'Supermarket No'}, inplace=True)
promotion_df.rename(columns={'code': 'Code', 'supermarkets': 'Supermarket No'}, inplace=True)
supermarkets_df.rename(columns={'supermarket_No': 'Supermarket No', 'Province':'province'}, inplace=True) # Added 'Province':'province' to fix the typo

# Merging DataFrames
merged_df = pd.merge(sales_df, items_df, on='Code', how='left')
merged_df = pd.merge(merged_df, promotion_df, on=['Code', 'Supermarket No'], how='left')
merged_df = pd.merge(merged_df, supermarkets_df, on='Supermarket No', how='left')

# Handling categorical variables using OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
# Replace 'sparse' with 'sparse_output'
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  
encoded_categories = encoder.fit_transform(merged_df[['type', 'province_y']])
encoded_df = pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out(['type', 'province_y']))
merged_df = pd.concat([merged_df, encoded_df], axis=1)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # Import numpy

# Visualizing the distribution of sales amounts
sns.histplot(merged_df['amount'], kde=True)
plt.title('Distribution of Sales Amounts')
plt.show()

# Filter the DataFrame to include only numeric columns for correlation calculation
numeric_df = merged_df.select_dtypes(include=[np.number]) # Now np is defined and accessible

# Correlation Heatmap
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder

# Selecting features and target
features = merged_df[['units', 'feature', 'display']]  
target = merged_df['amount']  

# Create a LabelEncoder object
encoder = LabelEncoder()

# Fit the encoder to the 'feature' column and transform it
features['feature_encoded'] = encoder.fit_transform(features['feature'])

# Drop the original 'feature' column
features = features.drop('feature', axis=1)

# Now features dataframe has numerical values for the 'feature' column

# ----> Encode the 'display' column to numerical values
features['display_encoded'] = encoder.fit_transform(features['display'])  
features = features.drop('display', axis=1)  # Drop the original 'display' column

# Now features dataframe has numerical values for the 'display' column

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Building the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

# Predicting and evaluating the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
