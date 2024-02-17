from utils import convert_column_headers, exclude_missing_negatives, exclude_invalid_products, calculate_metrics, flag_halloween_sales, cluster_underperforming_products, analyze_underperforming_products
import pandas as pd
import os
import matplotlib.pyplot as plt

"""Your objective is to identify the top 20 performing products and offer insightful answers to the following questions:

1. Unpacking Excellence:

Explore why these top 20 products consistently outperform the competition. Delve into the underlying factors driving their success, and don't hesitate to think creatively.

2. The Impact of Strategies and Events:

Investigate whether there are any specific strategies or events that have played a significant role in influencing the sales performance of these products. Look beyond the obvious and seek hidden catalysts."""



""" --------------------------------------------------------------------------------------------------------------------
Data Cleaning and Preprocessing"""
# Load inventory data
dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(dir_path, 'inventory.csv')
inventory_df = pd.read_csv(file_path)

# Convert Unix timestamp columns to human-readable dates
inventory_df = convert_column_headers(inventory_df)
# Exclude rows with more than 20% missing values
inventory_df = exclude_missing_negatives(inventory_df, 20)
# Exclude products with unreasonable inventory shifts
inventory_df = exclude_invalid_products(inventory_df, 1000)

# Interpolate missing values for numeric columns
numeric_cols = inventory_df.select_dtypes(include=['float64', 'int64']).columns
inventory_df[numeric_cols] = inventory_df[numeric_cols].apply(pd.to_numeric, errors='coerce').interpolate(method='linear', axis=0)

# Forward fill, backward fill to handle remaining missing values
inventory_df.ffill(axis=0, inplace=True)
inventory_df.bfill(axis=0, inplace=True)

""" --------------------------------------------------------------------------------------------------------------------
Identifying top 20"""

# Flag products with Halloween sales, these are the categories likely to have increased sales in my opinion
inventory_df = flag_halloween_sales(inventory_df, impacted_categories=['Gummy', 'Granola', 'Chocolate', 'Chips'])

# print(inventory_df.columns)

# Calculate metrics
sales_velocity, inventory_turnover_rate, replenishment_frequency, sustained_decreases = calculate_metrics(inventory_df.iloc[:, 4:]) # Inventory data starts from 5th column

# Combine metrics into a DataFrame for scoring
metrics_df = pd.DataFrame({
    'Sales Velocity': sales_velocity,
    'Inventory Turnover Rate': inventory_turnover_rate,
    'Replenishment Frequency': replenishment_frequency,
    'Sustained Decreases': sustained_decreases,
    'Halloween_Flag': inventory_df['Halloween_Flag']
})

# Rank products within each metric
ranks_df = metrics_df.rank(method='min', ascending=False)  # Lower rank = better performance

# Calculate a combined score
metrics_df['Combined Score'] = ranks_df.sum(axis=1)
metrics_df['Product'] = inventory_df['Product']

# Identify top 20 products based on the combined score
halloween_bonus = 10  # How much I think Halloween would affect rankings
metrics_df['Adjusted Score'] = metrics_df['Combined Score'] - (metrics_df['Halloween_Flag'] * halloween_bonus)
top_20_products = metrics_df.sort_values('Adjusted Score').head(20).merge(inventory_df[['Product', 'Category', 'BrandId']], on='Product', how='left')

top_categories = top_20_products.groupby('Category').size().sort_values(ascending=False)
top_brands = top_20_products.groupby('BrandId').size().sort_values(ascending=False)

print("Top Categories among Top 20 Products:\n", top_categories)
print("\nTop Brands among Top 20 Products:\n", top_brands)

print("Top 20 Performing Products:")
print(top_20_products)

""" --------------------------------------------------------------------------------------------------------------------
3. Recommendations for Improvement:

Extend your analysis to products that are currently underperforming. What actionable recommendations can you provide to enhance their sales and overall performance?"""

# Cluster underperforming products
underperforming_products, insights = cluster_underperforming_products(metrics_df)

# Analyze underperforming products
recommendations_df = analyze_underperforming_products(inventory_df, metrics_df)

# Print insights for each cluster
for cluster, insight in insights.items():
    print(f"Cluster {cluster}:")
    for k, v in insight.items():
        print(f"  {k}: {v}")
    print()
    
# Print recommendations for improvement, for now I only have one recommendation
print("Recommendations for Improvement:")
print(recommendations_df)

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