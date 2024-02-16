from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def convert_column_headers(df):
    """
    Converts DataFrame column names from Unix timestamps to human-readable dates
    for columns that contain Unix timestamps.

    Parameters:
    - df: pandas DataFrame containing the data.

    Returns:
    - DataFrame with converted column names.
    """
    new_columns = []
    for col in df.columns:
        try:
            # Convert to int, then to datetime, and finally to a date string
            readable_date = datetime.utcfromtimestamp(int(col)).strftime('%Y-%m-%d %H:%M:%S')
            new_columns.append(readable_date)
        except ValueError:
            # Keep the original column name if it's not a Unix timestamp
            new_columns.append(col)
    df.columns = new_columns
    return df

def calculate_daily_inventory_changes(df):
    """
    Calculates daily inventory changes for each product.
    
    Parameters:
    - df: DataFrame with inventory levels across dates in columns.
    
    Returns:
    - DataFrame with daily inventory changes.
    """
    # Assuming the first four columns are 'Unnamed: 0', 'Product', 'Category', 'BrandId', and the rest are dates
    inventory_levels = df.iloc[:, 4:]
    inventory_levels = inventory_levels.apply(pd.to_numeric, errors='coerce')
    daily_changes = inventory_levels.diff(axis=1)  # Calculate daily changes
    return daily_changes
