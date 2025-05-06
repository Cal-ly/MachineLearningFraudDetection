import pandas as pd
import numpy as np
import datetime
import os
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from pandas.api.types import is_numeric_dtype

import logging

# Configure logging
log_file = 'data_exploration/output/data_prep.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

# Data Preparation
logging.info("=== Data Preparation Started ===")
dataset_path = 'data/processed/cleaned_data.csv'
if os.path.exists(dataset_path):
    logging.info("Dataset found: %s", dataset_path)
else:
    logging.error("Dataset not found: %s", dataset_path)
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

# Load the dataset
dataset = pd.read_csv(dataset_path)
logging.info("Dataset loaded successfully. Shape: %s", dataset.shape)

target_column = 'isfraud'

features = ['transactionamt', 'productcd', 'card1', 'card2', 'card3', 'card5', 'card6', 'addr1', 'dist1', 'p_emaildomain', 'r_emaildomain',
              'c1', 'c2', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14',
              'v62', 'v70', 'v76', 'v78', 'v82', 'v91', 'v127', 'v130', 'v139', 'v160', 'v165', 'v187', 'v203',
              'v207', 'v209', 'v210', 'v221', 'v234', 'v257', 'v258', 'v261', 'v264', 'v266', 'v267', 'v271', 'v274',
              'v277', 'v283', 'v285', 'v289', 'v291', 'v294', 'id_01', 'id_02', 'id_05', 'id_06', 'id_09', 'id_13', 'id_17', 'id_19', 'id_20', 'devicetype', 'deviceinfo']

# Split the datasat into train and test sets chronologically, assuming 'transactiondt' is the datetime column in the dataset
train_data = dataset[dataset['transactiondt'] < dataset['transactiondt'].quantile(0.8)][features + [target_column]]
test_data = dataset[dataset['transactiondt'] >= dataset['transactiondt'].quantile(0.8)][features + [target_column]]
logging.info("Train data shape: %s", train_data.shape)
logging.info("Test data shape: %s", test_data.shape)

# Separate features and target variable
X_train = train_data[features]
y_train = train_data[target_column]
X_test = test_data[features]
y_test = test_data[target_column]

# Handle missing values by filling with mean for numeric columns - This could be improved with more sophisticated imputation methods?
for col in X_train.columns:
    if is_numeric_dtype(X_train[col]):
        X_train.loc[:, col] = X_train[col].fillna(X_train[col].mean())
        X_test.loc[:, col] = X_test[col].fillna(X_test[col].mean())

# Convert categorical variables to numeric using Label Encoding
for col in X_train.select_dtypes(include=['object']).columns:
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[col] = ordinal_encoder.fit_transform(X_train[[col]])
    X_test[col] = ordinal_encoder.transform(X_test[[col]])

# Ensure all columns are numeric
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# See if the target variable is imbalanced
fraud_count = y_train.value_counts()
logging.info("Target variable distribution:\n%s", fraud_count)
logging.info("Percentage of fraud cases: %.2f%%", (fraud_count[1] / len(y_train)) * 100)

# Train the model using XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1, enable_categorical=True)
model.fit(X_train, y_train)
logging.info("Model training completed.")

# Save the model
import joblib
model_path = 'data/models/xgboost_model.pkl'
if not os.path.exists('data/models'):
    os.makedirs('data/models')
    logging.info("Model directory created: %s", 'data/models')

# Save the model using joblib
joblib.dump(model, model_path)
logging.info("Model saved to %s", model_path)
