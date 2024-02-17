from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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

def exclude_missing_negatives(df, threshold=20):
    """
    Excludes rows from a DataFrame where the percentage of missing values exceeds a specified threshold
    and also removes rows containing any negative inventory values.

    Parameters:
    - df: pandas DataFrame.
    - threshold: int or float, the percentage threshold of missing values to exclude a row.

    Returns:
    - DataFrame with rows having missing values below the threshold and no negative values.
    """
    # Calculate the percentage of missing values for each row
    percent_missing_per_row = df.isnull().mean(axis=1) * 100
    
    # Filter rows where the percentage of missing is less than the threshold
    filtered_df = df[percent_missing_per_row < threshold].copy()
    
    # Assuming inventory data starts from the 5th column
    inventory_data_columns = filtered_df.columns[4:]
    
    # Convert inventory data columns to numeric, errors='coerce' will turn invalid parsing into NaN
    for col in inventory_data_columns:
        filtered_df.loc[:, col] = pd.to_numeric(filtered_df[col], errors='coerce')
    
    # Filter out rows with any negative values in inventory data columns
    filtered_df = filtered_df.loc[(filtered_df[inventory_data_columns] >= 0).all(axis=1)]
    
    return filtered_df


def exclude_invalid_products(df, threshold=1000):
    """
    Excludes products from the inventory that have shifts exceeding a specified threshold.

    Parameters:
    - df: pandas DataFrame with inventory levels across dates in columns.
    - threshold: int, the threshold for unreasonable inventory shifts.

    Returns:
    - DataFrame with products not having unreasonable inventory shifts.
    """
    # Calculate inventory changes between consecutive timestamps for each product
    inventory_levels = df.iloc[:, 4:].apply(pd.to_numeric, errors='coerce')  # Assuming inventory levels start at 5th column
    inventory_changes = inventory_levels.diff(axis=1)
    
    # Identify products with any changes exceeding the threshold
    unreasonable_shifts = (inventory_changes.abs() > threshold).any(axis=1)
    
    # Exclude these products from the dataset
    filtered_df = df[~unreasonable_shifts]
    
    return filtered_df

    
def flag_halloween_sales(df, start_date='2024-10-24', end_date='2024-10-29', impacted_categories=None):
    """
    Adds a Halloween impact flag to products based on categories likely to have increased sales.
    For simplicity, I've only implemented this function for Halloween as the dates provided in the data frame range 
    from September 9 to October 29. This means I'm only considering the impact of Halloween sales on the days
    leading up to the October 31.

    Parameters:
    - df: DataFrame with inventory data and datetime columns as headers.
    - start_date: The beginning date of the Halloween impact period.
    - end_date: The end date of the Halloween impact period.
    - impacted_categories: List of product categories expected to be impacted by Halloween.

    Returns:
    - DataFrame with a new boolean column 'Halloween_Flag' for each date column within the Halloween period.
    """
    if impacted_categories is None:
        impacted_categories = ['Gummy', 'Granola', 'Chocolate', 'Chips']  # Categories likely to be affected by Halloween

    # Convert the start and end dates to pandas datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Identify date columns by trying to convert them to datetime objects
    date_columns = [col for col in df.columns if pd.to_datetime(col, errors='ignore').__class__ == pd.Timestamp]
    
    # Initialize the Halloween_Flag column to False
    df['Halloween_Flag'] = False

    # Flag products within the impacted categories and date range
    for col in date_columns:
        date = pd.to_datetime(col)
        if start_date <= date <= end_date:
            df.loc[df['Category'].isin(impacted_categories), 'Halloween_Flag'] = True

    return df

def calculate_metrics(df):
    """ Calculates sales velocity, inventory turnover rate, replenishment frequency, and sustained decreases."""
    
    inventory_changes = df.diff(axis=1).fillna(0)  # Calculate daily inventory changes
    
    # Sales velocity, assumes negative trend indicates a sale 
    sales_velocity = inventory_changes[inventory_changes < 0].abs().sum(axis=1) / df.shape[1]
    
    # Inventory turnover rate (total sales divided by average inventory level), just a simple approximation
    total_sales = inventory_changes[inventory_changes < 0].abs().sum(axis=1)
    average_inventory = df.mean(axis=1)
    inventory_turnover_rate = total_sales / average_inventory.replace(0, np.nan)  # Avoid division by zero
    
    # Replenishment frequency, count of days with positive inventory changes (replenishments)
    replenishment_frequency = (inventory_changes > 0).sum(axis=1)
    
    # Sustained decreases, longest consecutive period of inventory decreases
    sustained_decreases = inventory_changes.apply(lambda x: (x < 0).astype(int).groupby(x.ne(x.shift()).cumsum()).cumcount().max(), axis=1)
    
    return sales_velocity, inventory_turnover_rate, replenishment_frequency, sustained_decreases

def cluster_underperforming_products(df, n_clusters=5):
    """
    Identifies underperforming products and clusters them to provide insights for improvement.
    
    Parameters:
    - df: DataFrame containing product performance metrics.
    - n_clusters: Number of clusters to form.
    
    Returns:
    - df: DataFrame with an additional 'Cluster' column indicating the assigned cluster for each product.
    - insights: Dictionary with insights for each cluster.
    """
    # Assuming 'Sales Velocity' and 'Inventory Turnover Rate' are performance indicators
    # Filter out underperforming products (you can define your own criteria here)
    underperforming_df = df[(df['Sales Velocity'] < df['Sales Velocity'].median()) &
                            (df['Inventory Turnover Rate'] < df['Inventory Turnover Rate'].median())]
    
    # Extract features for clustering
    features = underperforming_df[['Sales Velocity', 'Inventory Turnover Rate']].values
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    underperforming_df['Cluster'] = kmeans.fit_predict(features_scaled)
    
    # Analyze clusters for insights
    insights = {}
    for cluster in range(n_clusters):
        cluster_products = underperforming_df[underperforming_df['Cluster'] == cluster]
        insights[cluster] = {
            'Average Sales Velocity': cluster_products['Sales Velocity'].mean(),
            'Average Inventory Turnover Rate': cluster_products['Inventory Turnover Rate'].mean(),
            # Could possibly add more metrics but for now, this is all I've implemented
        }
    
    return underperforming_df, insights

def analyze_underperforming_products(df, metrics):
    """
    Enhance the function to handle NaN values and correctly modify the DataFrame.
    """
    # Impute NaN values in your features (was receiving errors with the NaN values, so just impute)
    imputer = SimpleImputer(strategy='mean')
    features = metrics[['Sales Velocity', 'Inventory Turnover Rate', 'Replenishment Frequency', 'Sustained Decreases']].values
    features_imputed = imputer.fit_transform(features)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(features_imputed)
    
    # Safely add the cluster information to your metrics DataFrame
    metrics['Cluster'] = clusters
    
    # Identify underperforming products (You might need to adapt this based on your criteria)
    underperforming_products = metrics[metrics['Sales Velocity'] <= metrics['Sales Velocity'].quantile(0.25)]
    
    # Provide recs for improvement. for simplicity, I just have one recommendation, but could add on
    underperforming_products = underperforming_products.copy() # Attemps to solve SettingWithCopyWarning
    underperforming_products['Recommendations'] = 'Review pricing or marketing strategies.'
    
    return underperforming_products[['Product', 'Sales Velocity', 'Inventory Turnover Rate', 'Recommendations']]