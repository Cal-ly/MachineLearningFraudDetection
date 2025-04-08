import pandas as pd
import numpy as np
import datetime
import os
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from pandas.api.types import is_numeric_dtype

import logging

# Configure logging
log_file = 'data_exploration/output/data_prep.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

