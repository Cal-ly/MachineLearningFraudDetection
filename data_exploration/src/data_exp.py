import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

# Configure logging
log_file = 'data_exploration/output/data_exploration.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

# Data Exploration
logging.info("=== Data Exploration Started ===")

# Load the datasets
identity_path = 'data/raw/train_identity.csv'
transaction_path = 'data/raw/train_transaction.csv'

identity_data = pd.read_csv(identity_path)
transaction_data = pd.read_csv(transaction_path)

# Merge the datasets on 'TransactionID' and log the shape of the merged data
merged_data = pd.merge(transaction_data, identity_data, on='TransactionID', how='left')
logging.info("Merged Data shape: %s", merged_data.shape)

# Convert the column names to lowercase and remove dash etc.
merged_data.columns = merged_data.columns.str.lower()
merged_data.columns = merged_data.columns.str.replace("-", "_")
merged_data = merged_data.sort_values(by='transactiondt')

# Log the first few rows of the merged data
logging.info("First few rows of the merged data:\n%s", merged_data.head().to_string())

# Log the data types of each column
logging.info("Data types of each column:\n%s", merged_data.dtypes)

# Log the number of missing values in each column
logging.info("Number of missing values in each column:\n%s", merged_data.isnull().sum())

# Log basic statistics of the merged data
logging.info("Basic statistics of the merged data:\n%s", merged_data.describe())

logging.info("=== Data Exploration Completed ===")

# Save the cleaned data to a new CSV file
cleaned_data_path = 'data/processed/cleaned_data.csv'
merged_data.to_csv(cleaned_data_path, index=False)
logging.info("Cleaned data saved to %s", cleaned_data_path)