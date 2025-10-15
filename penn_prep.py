import pandas as pd

# Load the Excel file
file_path = 'cleaned_V11.csv'  # Replace with your file path

# Read the Excel file
df = pd.read_csv(file_path)

# Specify the columns
country_col = 'countrycode' # Replace with your country column name
value_cols = ['rgdpe', 'rgdpo', 'pop'] # Replace with your value column name

# Function to calculate percentage change
def calculate_percentage_change(df, group, cols):
    df = df.copy()
    for value_col in value_cols:
        name = f'{value_col}_pct_change'
        df[name] = df.groupby(country_col)[value_col].pct_change()
    return df

# Apply the function to the DataFrame
df = calculate_percentage_change(df, country_col, value_cols)
print(df.head())