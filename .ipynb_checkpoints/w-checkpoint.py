import pandas as pd

# Load the data from a text file
data = pd.read_csv('your_data.txt')

# Explore the data
print(data.head())  # View the first few rows
print(data.info())  # Get information about the data

# Handling missing values
data = data.dropna()  # Drop rows with any NaN values

# Remove duplicates
data = data.drop_duplicates()

# Correct data types
# Assuming 'Rating' is a column you want to convert to integers
data['Rating'] = data['Rating'].astype(int)

# Standardize/Normalize (if needed)
# Example: Standardize the 'Rating' column
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[['Rating']] = scaler.fit_transform(data[['Rating']])

# Save cleaned data
data.to_csv('cleaned_data.txt', index=False, sep='\t')  # You can specify the delimiter if needed

