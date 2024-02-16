from utils import convert_column_headers, calculate_daily_inventory_changes
import pandas as pd
import os
import matplotlib.pyplot as plt

"""Your objective is to identify the top 20 performing products and offer insightful answers to the following questions:

1. Unpacking Excellence:

Explore why these top 20 products consistently outperform the competition. Delve into the underlying factors driving their success, and don't hesitate to think creatively.

2. The Impact of Strategies and Events:

Investigate whether there are any specific strategies or events that have played a significant role in influencing the sales performance of these products. Look beyond the obvious and seek hidden catalysts.

3. Recommendations for Improvement:

Extend your analysis to products that are currently underperforming. What actionable recommendations can you provide to enhance their sales and overall performance?"""

# Get the directory where the script is located
dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(dir_path, 'inventory.csv')
inventory_df = pd.read_csv(file_path)

# Convert Unix timestamp columns to human-readable dates
inventory_df = convert_column_headers(inventory_df)

# Interpolate missing values for numeric columns
numeric_cols = inventory_df.select_dtypes(include=['float64', 'int64']).columns
inventory_df[numeric_cols] = inventory_df[numeric_cols].apply(pd.to_numeric, errors='coerce').interpolate(method='linear', axis=0)

# Forward fill, backward fill to handle remaining missing values
inventory_df.ffill(axis=0, inplace=True)
inventory_df.bfill(axis=0, inplace=True)

# print(inventory_df.head())

# Calculate daily inventory changes
daily_changes = calculate_daily_inventory_changes(inventory_df)

# Aggregate inventory changes to score products
total_decrease = daily_changes[daily_changes < 0].sum(axis=1)  # Total decrease per product
replenishment_count = (daily_changes > 0).sum(axis=1)  # Count of days with inventory increase

# Simple scoring: Sum of total decrease (in magnitude) and replenishment count
product_scores = total_decrease.abs() + replenishment_count

# Identify top 20 products based on scores
top_20_indices = product_scores.nlargest(20).index
top_20_products = inventory_df.loc[top_20_indices, 'Product']

print("Top 20 Performing Products:")
print(top_20_products)
# Check data types and missing values
# print(inventory_df.info())

# Get summary statistics for numerical columns
# print(inventory_df.describe())

# Check for missing values
# print(inventory_df.isnull().sum())

# Check data types
# print(inventory_df.dtypes)

# Check for duplicates
# print(inventory_df.duplicated().sum())

# Summary statistics to identify outliers
# print(inventory_df.describe())

# plot_distribution(inventory_df, 'sales_volume', 'Distribution of Sales Volume', 'Sales Volume', 'Frequency')
# plot_time_series(inventory_df, 'date_column_name', 'sales_volume', 'Sales Volume Over Time', 'Date', 'Sales Volume')
