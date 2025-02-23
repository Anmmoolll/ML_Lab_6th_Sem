import pandas as pd

# Step 1: Import sales data from a CSV file
input_file = "sales_data.csv"  # Replace with actual file path
sales_df = pd.read_csv(input_file)

# Step 2: Perform basic data cleaning
sales_df.dropna(inplace=True)  # Remove rows with missing values

# Step 3: Add a new column for Sales Tax (assuming 10% tax)
sales_df['Sales Tax'] = sales_df['Sales Amount'] * 0.10

# Step 4: Export the updated data to a new CSV file
output_file = "cleaned_sales_data.csv"
sales_df.to_csv(output_file, index=False)

print("Data successfully cleaned and exported to:", output_file)
