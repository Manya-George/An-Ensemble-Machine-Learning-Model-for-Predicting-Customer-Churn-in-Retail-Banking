import mysql.connector
from core.config import DB_CONFIG, MODEL_PATH
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import joblib

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def get_model_artifacts_info():
    """Return dict with model status and stored metrics (if saved in joblib)."""
    try:
        from core.ml_engine import MODEL  # Import here to avoid circular dependency
        
        info = {
            'loaded': MODEL is not None,
            'model_path': MODEL_PATH if MODEL_PATH else None,
            'file_exists': False,
            'file_mtime_iso': None,
            'metrics': None
        }
        path = info['model_path']
        if path and os.path.exists(path):
            info['file_exists'] = True
            mtime = os.path.getmtime(path)
            # Deprecation note: convert to timezone-aware in future
            info['file_mtime_iso'] = datetime.utcfromtimestamp(mtime).isoformat() + 'Z'
            try:
                artifacts = joblib.load(path)
                if isinstance(artifacts, dict):
                    metrics = artifacts.get('metrics') or artifacts.get('model_metrics') or artifacts.get('training_metrics')
                    info['metrics'] = metrics
            except Exception as e:
                print(f"get_model_artifacts_info: failed reading joblib: {e}")
        return info
    except Exception as e:
        print(f"get_model_artifacts_info error: {e}")
        from core.ml_engine import MODEL
        return {'loaded': MODEL is not None}

def to_python(val):
    """Convert numpy/pandas types to pure Python types for MySQL."""
    if isinstance(val, (np.integer, np.int64)): 
        return int(val)
    if isinstance(val, (np.floating, np.float64)): 
        return float(val)
    if isinstance(val, (np.bool_,)): 
        return bool(val)
    if isinstance(val, (pd.Timestamp,)):
        return str(val)
    return val

def calculate_risk_level(probability):
    """
    Determine risk level based on the dynamic optimal threshold.
    FIXED: Import THRESHOLD from ml_engine to ensure consistency
    """
    try:
        from core.ml_engine import THRESHOLD
    except ImportError:
        THRESHOLD = 0.5  # Fallback
    
    # High risk: above optimal threshold
    if probability >= THRESHOLD:
        return 'high'
    # Medium risk: 75% of the way between 0.5 and threshold
    elif probability >= (0.5 + (THRESHOLD - 0.5) * 0.75): 
        return 'medium'
    # Low risk: below medium threshold
    else:
        return 'low'