import joblib
import os
import pandas as pd
import logging
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Configure logging
log_file = 'data_exploration/output/model_eval_dalia.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("=== Model Training and Evaluation Started ===")

# Load the prepped datasets
train_data_path = 'data/processed/train_data_dalia.csv'
test_data_path = 'data/processed/test_data_dalia.csv'

if os.path.exists(train_data_path) and os.path.exists(test_data_path):
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    logging.info("Train and test datasets loaded successfully.")
else:
    logging.error("Prepped datasets not found.")
    raise FileNotFoundError("Prepped datasets not found.")

# Define the target variable and features
target_column = 'isfraud'
features = [col for col in train_data.columns if col != target_column]

# Separate features and target variable
X_train = train_data[features]
y_train = train_data[target_column]
X_test = test_data[features]
y_test = test_data[target_column]

# Train the XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
logging.info("Training the XGBoost model...")
model.fit(X_train, y_train)
logging.info("Model training completed.")

# Save the trained model
model_path = 'data/models/xgboost_model_dalia.pkl'
if not os.path.exists('data/models'):
    os.makedirs('data/models')
joblib.dump(model, model_path)
logging.info("Model saved to %s", model_path)

# Evaluate the model
logging.info("Evaluating the model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Compute evaluation metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Log the results
logging.info("ROC AUC Score: %.4f", roc_auc)
logging.info("Confusion Matrix:\n%s", conf_matrix)
logging.info("Classification Report:\n%s", class_report)

# Print results to console
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)




