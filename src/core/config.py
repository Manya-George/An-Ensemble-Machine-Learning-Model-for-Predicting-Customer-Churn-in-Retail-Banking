import os
from dotenv import load_dotenv
import os
from datetime import datetime

APP_START_TIME = datetime.utcnow()

load_dotenv()

MODEL_PATH = 'model_artifacts/churn_ensemble_deploy.joblib' 
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "database": os.getenv("DB_NAME"),
    "port": os.getenv("DB_PORT", "3306")
}

EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email': os.getenv('EMAIL_ADDRESS'),
    'password': os.getenv('EMAIL_PASSWORD')
}

SESSION_CONFIG = {
    'SESSION_TYPE': 'filesystem',
    'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_SECURE': False,
    'SESSION_COOKIE_SAMESITE': 'Lax'
}
