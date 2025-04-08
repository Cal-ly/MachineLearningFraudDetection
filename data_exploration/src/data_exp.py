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

# Set target variable and features
target_column = 'isfraud'
features = ['transactionamt', 'productcd', 'card1', 'card2', 'card3', 'card5', 'card6', 'addr1', 'dist1', 'p_emaildomain', 'r_emaildomain',
              'c1', 'c2', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14',
              'v62', 'v70', 'v76', 'v78', 'v82', 'v91', 'v127', 'v130', 'v139', 'v160', 'v165', 'v187', 'v203',
              'v207', 'v209', 'v210', 'v221', 'v234', 'v257', 'v258', 'v261', 'v264', 'v266', 'v267', 'v271', 'v274',
              'v277', 'v283', 'v285', 'v289', 'v291', 'v294', 'id_01', 'id_02', 'id_05', 'id_06', 'id_09', 'id_13', 'id_17', 'id_19', 'id_20', 'devicetype', 'deviceinfo']

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