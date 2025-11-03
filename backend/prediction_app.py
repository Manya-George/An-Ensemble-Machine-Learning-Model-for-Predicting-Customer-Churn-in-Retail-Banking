# prediction_app.py
# Updated: session management + Flask-Session (filesystem), login_required decorator
# - Keeps all original routes/logic and debug prints
# - Adds PERMANENT_SESSION_LIFETIME and uses session.permanent = True on login success
# - Adds @login_required decorator for protected routes (optional usage)
# - Adds /api/mark-otp-expired endpoint
# NOTE: Make sure flask-session is installed: pip install Flask-Session

from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from flask import g
from flask_session import Session
import mysql.connector
import bcrypt
import random
import string
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import joblib
import uuid
import shap
import traceback
import time
from functools import wraps

# Application start time (used for uptime)
APP_START_TIME = datetime.utcnow()

load_dotenv()

app = Flask(__name__, static_folder=None)
# Secret key - set in .env for production
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this')

# Session config - using filesystem sessions for local development
# Install Flask-Session: pip install Flask-Session
app.config['SESSION_TYPE'] = 'filesystem'
# Optional: directory for session files
SESSION_FILE_DIR = os.getenv('SESSION_FILE_DIR', None)
if SESSION_FILE_DIR:
    os.makedirs(SESSION_FILE_DIR, exist_ok=True)
    app.config['SESSION_FILE_DIR'] = SESSION_FILE_DIR

# Keep cookies accessible to frontend during local dev
app.config['SESSION_COOKIE_HTTPONLY'] = True
# Locally over http (not secure). Set to True in production and use HTTPS.
app.config['SESSION_COOKIE_SECURE'] = False
# Set SameSite depending on your frontend host; 'Lax' works for many.
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Lifetime for session (inactive timeout)
app.permanent_session_lifetime = timedelta(minutes=30)  # change as you prefer

# Initialize Flask-Session
Session(app)

# Allow cross origin for API routes (frontend may run on different origin)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# Middleware: log requests into api_metrics table (non-blocking best-effort)
@app.before_request
def log_api_request():
    try:
        # Avoid logging static assets or favicon; only log API endpoints
        path = request.path
        if not path.startswith('/api'):
            return
        method = request.method
        timestamp = datetime.utcnow()
        user_id = session.get('user_id') if session.get('user_id') else None

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO api_metrics (path, method, userID, timestamp)
                VALUES (%s, %s, %s, %s)
            """, (path, method, user_id, timestamp))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            # If insertion fails, print to console and continue
            print(f"api_metrics insert error: {e}")
    except Exception as e:
        print(f"request logging error: {e}")

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MODEL_PATH = 'catboost_churn_model.joblib'

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database connection configuration from environment
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "database": os.getenv("DB_NAME"),
    "port": os.getenv("DB_PORT", "3306")
}

# Email configuration
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email': os.getenv('EMAIL_ADDRESS', 'your-email@gmail.com'),
    'password': os.getenv('EMAIL_PASSWORD', 'your-app-password')
}

# Load the trained model & artifacts (if available)
MODEL = None
PREPROCESSOR = None
SELECTOR = None
EXPLAINER = None
FEATURE_ORIGIN_MAP = {}
SELECTED_FEATURES = []
PROCESSED_FEATURE_NAMES = []

try:
    model_artifacts = joblib.load(MODEL_PATH)
    MODEL = model_artifacts.get('model', None)
    PREPROCESSOR = model_artifacts.get('preprocessor', None)
    SELECTOR = model_artifacts.get('selector', None)

    EXPLAINER = model_artifacts.get('explainer', None)
    FEATURE_ORIGIN_MAP = model_artifacts.get('feature_origin_map', {}) or model_artifacts.get('feature_map', {})
    SELECTED_FEATURES = model_artifacts.get('selected_features') \
                        or (model_artifacts.get('feature_names') and model_artifacts['feature_names'].get('selected')) \
                        or model_artifacts.get('feature_names', {}).get('selected') or []
    PROCESSED_FEATURE_NAMES = model_artifacts.get('processed_feature_names') or model_artifacts.get('feature_names', {}).get('processed', [])

    if EXPLAINER is None and MODEL is not None:
        try:
            EXPLAINER = shap.TreeExplainer(MODEL)
        except Exception as e:
            print("Warning: Could not initialize SHAP explainer from model:", e)
            EXPLAINER = None

    print("✅ Churn prediction model loaded successfully!")
except Exception as e:
    print(f"⚠️  Warning: Could not load model artifacts from {MODEL_PATH}: {e}")
    MODEL = None
    PREPROCESSOR = None
    SELECTOR = None
    EXPLAINER = None
    FEATURE_ORIGIN_MAP = {}
    SELECTED_FEATURES = []
    PROCESSED_FEATURE_NAMES = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def get_model_artifacts_info():
    """Return dict with model status and stored metrics (if saved in joblib)."""
    try:
        info = {
            'loaded': MODEL is not None,
            'model_path': MODEL_PATH if 'MODEL_PATH' in globals() else None,
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
        return {'loaded': MODEL is not None}

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

def send_otp_email(email, otp_code, username):
    """Send OTP via email"""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['email']
        msg['To'] = email
        msg['Subject'] = 'Your OTP Code - Banking System'
        
        body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; padding: 20px;">
                <h2>OTP Verification</h2>
                <p>Hello {username},</p>
                <p>Your One-Time Password (OTP) for login verification is:</p>
                <h1 style="color: #2196F3; letter-spacing: 5px;">{otp_code}</h1>
                <p>This OTP will expire in 1 minute.</p>
                <p>If you didn't request this code, please ignore this email.</p>
                <br>
                <p>Best regards,<br>Banking System Team</p>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['email'], EMAIL_CONFIG['password'])
        server.send_message(msg)
        server.quit()
        
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def calculate_risk_level(probability):
    """Determine risk level based on churn probability"""
    if probability >= 0.7:
        return 'high'
    elif probability >= 0.4:
        return 'medium'
    else:
        return 'low'

# ----------------------------
# System logging helpers
# ----------------------------
def log_system_event(endpoint, method, description, status_code=200, user_id=None):
    """
    Save a log entry to the system_logs table.
    This function is robust: it will print a warning if logging fails but won't raise.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO system_logs (userID, endpoint, method, action_description, status_code)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (user_id, endpoint, method, description, status_code)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"[Logging Error] Could not save log (endpoint={endpoint}): {e}")

@app.after_request
def after_request(response):
    """
    After each request, automatically log it to the system_logs table.
    We extract a brief description from the JSON response if available.
    """
    try:
        user_id = session.get('user_id') if session else None
        endpoint = request.path
        method = request.method
        status = response.status_code
        desc = ""
        try:
            if response.is_json:
                j = response.get_json(silent=True)
                if isinstance(j, dict):
                    desc = j.get('message') or j.get('msg') or j.get('error') or str(j)[:500]
                else:
                    desc = str(j)[:500]
        except Exception:
            desc = ""
        log_system_event(endpoint, method, desc, status, user_id)
    except Exception as e:
        print("Warning: after_request logging failed:", e)
    return response

# ----------------------------
# Session / Auth helpers
# ----------------------------
def login_required(fn):
    """
    Decorator to enforce authentication. Returns 401 if not authenticated.
    Use on routes that require login.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        return fn(*args, **kwargs)
    return wrapper

# ============== AUTHENTICATION ROUTES ==============

@app.route('/api/login', methods=['POST'])
def login():
    """Handle user login"""
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            res = jsonify({'success': False, 'message': 'Username and password required'}), 400
            return res
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(
            "SELECT userID, username, email, password_hash, role, IFNULL(is_active, 1) AS is_active FROM users WHERE username = %s",
            (username,)
        )
        user = cursor.fetchone()
        
        if not user:
            cursor.close()
            conn.close()
            res = jsonify({'success': False, 'message': 'Invalid username or password'}), 401
            return res

        # Check active flag (if your users table has is_active column)
        if 'is_active' in user and (user.get('is_active') in (0, '0', False)):
            cursor.close()
            conn.close()
            try:
                log_system_event("/api/login", "POST", f"Unauthorized login attempt for {username}", 401, user_id=None)
            except:
                pass
            return jsonify({'success': False, 'message': 'Unauthorized user login attempt'}), 403
        
        if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            cursor.close()
            conn.close()
            res = jsonify({'success': False, 'message': 'Invalid username or password'}), 401
            return res
        
        otp_code = generate_otp()
        expires_at = datetime.now() + timedelta(minutes=1)
        
        cursor.execute(
            "INSERT INTO otps (userID, otp_code, expiresAt) VALUES (%s, %s, %s)",
            (user['userID'], otp_code, expires_at)
        )
        conn.commit()
        
        email_parts = user['email'].split('@')
        masked_email = f"{email_parts[0][0]}{'*' * (len(email_parts[0]) - 1)}@{email_parts[1]}"
        
        email_sent = send_otp_email(user['email'], otp_code, user['username'])
        
        if not email_sent:
            cursor.close()
            conn.close()
            res = jsonify({'success': False, 'message': 'Failed to send OTP email'}), 500
            return res
        
        # Save temp information in session (temp_user_id used until OTP is verified)
        session.permanent = True  # respects app.permanent_session_lifetime
        session['temp_user_id'] = user['userID']
        session['username'] = user['username']
        session['role'] = user['role']
        # Do NOT set authenticated until OTP is verified.
        
        cursor.close()
        conn.close()
        
        # manual log for login OTP sent
        try:
            log_system_event("/api/login", "POST", f"OTP sent to {user['username']}", 200, user_id=user['userID'])
        except:
            pass
        
        return jsonify({
            'success': True,
            'message': 'OTP sent successfully',
            'masked_email': masked_email
        }), 200
        
    except Exception as e:
        print(f"Login error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error'}), 500

@app.route('/api/verify-otp', methods=['POST'])
def verify_otp():
    """Verify OTP code"""
    try:
        data = request.json
        otp_code = data.get('otp')
        
        if not otp_code or len(otp_code) != 6:
            return jsonify({'success': False, 'message': 'Invalid OTP format'}), 400
        
        user_id = session.get('temp_user_id')
        if not user_id:
            return jsonify({'success': False, 'message': 'Session expired. Please login again'}), 401
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(
            """SELECT otpID, otp_code, expiresAt, isUsed 
               FROM otps 
               WHERE userID = %s AND isUsed = FALSE 
               ORDER BY createdAt DESC 
               LIMIT 1""",
            (user_id,)
        )
        otp_record = cursor.fetchone()
        
        if not otp_record:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'No valid OTP found'}), 401
        
        if datetime.now() > otp_record['expiresAt']:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'OTP has expired'}), 401
        
        if otp_code != otp_record['otp_code']:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Invalid OTP'}), 401
        
        cursor.execute(
            "UPDATE otps SET isUsed = TRUE WHERE otpID = %s",
            (otp_record['otpID'],)
        )
        conn.commit()
        
        # Promote to authenticated user
        session['user_id'] = user_id
        session['authenticated'] = True
        session.permanent = True  # extend lifetime on successful auth
        session.pop('temp_user_id', None)
        
        role = session.get('role')
        username = session.get('username')
        
        cursor.close()
        conn.close()
        
        try:
            log_system_event("/api/verify-otp", "POST", f"OTP verified for user {username}", 200, user_id=user_id)
        except:
            pass
        
        return jsonify({
            'success': True,
            'message': 'OTP verified successfully',
            'role': role,
            'username': username
        }), 200
        
    except Exception as e:
        print(f"OTP verification error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error'}), 500

@app.route('/api/resend-otp', methods=['POST'])
def resend_otp():
    """Resend OTP to user"""
    try:
        user_id = session.get('temp_user_id')
        if not user_id:
            return jsonify({'success': False, 'message': 'Session expired. Please login again'}), 401
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(
            "SELECT username, email FROM users WHERE userID = %s",
            (user_id,)
        )
        user = cursor.fetchone()
        
        if not user:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'User not found'}), 404
        
        otp_code = generate_otp()
        expires_at = datetime.now() + timedelta(minutes=1)
        
        cursor.execute(
            "INSERT INTO otps (userID, otp_code, expiresAt) VALUES (%s, %s, %s)",
            (user_id, otp_code, expires_at)
        )
        conn.commit()
        
        email_sent = send_otp_email(user['email'], otp_code, user['username'])
        
        cursor.close()
        conn.close()
        
        if not email_sent:
            return jsonify({'success': False, 'message': 'Failed to send OTP email'}), 500

        try:
            log_system_event("/api/resend-otp", "POST", f"OTP resent to {user['username']}", 200, user_id=user_id)
        except:
            pass
        
        return jsonify({
            'success': True,
            'message': 'OTP resent successfully'
        }), 200
        
    except Exception as e:
        print(f"Resend OTP error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error'}), 500

@app.route('/api/mark-otp-expired', methods=['POST'])
def mark_otp_expired():
    """Mark latest OTP as used/expired (called from frontend when timer runs out)"""
    try:
        user_id = session.get('temp_user_id')
        if not user_id:
            return jsonify({'success': False, 'message': 'No temporary user in session'}), 400
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE otps SET isUsed = TRUE WHERE userID=%s AND isUsed = FALSE ORDER BY createdAt DESC LIMIT 1
        """, (user_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'success': True, 'message': 'OTP marked as expired'}), 200
    except Exception as e:
        print(f"mark_otp_expired error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error'}), 500

# ----------------------------
# Helper functions for SHAP & features
# ----------------------------
def _get_processed_and_selected_feature_names(preprocessor, selector):
    processed_names = []
    selected_names = []
    try:
        num_cols = []
        cat_cols = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'num':
                num_cols = list(cols)
            elif name == 'cat':
                cat_cols = list(cols)
        ohe = preprocessor.named_transformers_.get('cat').named_steps.get('ohe')
        if ohe is not None:
            if hasattr(ohe, "get_feature_names_out"):
                ohe_names = list(ohe.get_feature_names_out(cat_cols))
            else:
                ohe_names = list(ohe.get_feature_names(cat_cols))
        else:
            ohe_names = []
        processed_names = list(num_cols) + ohe_names
        if selector is not None:
            mask = selector.get_support()
            selected_names = [n for n, keep in zip(processed_names, mask) if keep]
        else:
            selected_names = processed_names
    except Exception as e:
        print("Warning reconstructing feature names:", e)
        try:
            if selector is not None:
                num_selected = int(selector.transform(np.zeros((1, selector.estimator_.coef_.shape[1] if hasattr(selector.estimator_, 'coef_') else 1))).shape[1])
                selected_names = [f"feature_{i}" for i in range(num_selected)]
                processed_names = [f"feature_{i}" for i in range(len(selected_names))]
            else:
                processed_names = [f"feature_{i}" for i in range(1)]
                selected_names = processed_names
        except Exception:
            processed_names = [f"feature_{i}" for i in range(1)]
            selected_names = processed_names
    return processed_names, selected_names

def _map_processed_to_original(processed_name):
    if FEATURE_ORIGIN_MAP and processed_name in FEATURE_ORIGIN_MAP:
        return FEATURE_ORIGIN_MAP[processed_name]
    if isinstance(processed_name, str) and '_' in processed_name:
        return processed_name.split('_')[0]
    return processed_name

def _get_top_features_for_selected_row(shap_row_values, selected_feature_names, top_n=3):
    try:
        abs_vals = np.abs(shap_row_values)
        if abs_vals.ndim == 1:
            top_idx = np.argsort(abs_vals)[-top_n:][::-1]
        else:
            flat = abs_vals.flatten()
            top_idx = np.argsort(flat)[-top_n:][::-1]
        top_processed = [selected_feature_names[i] for i in top_idx if i < len(selected_feature_names)]
        mapped = [_map_processed_to_original(p) for p in top_processed]
        seen = set()
        dedup = []
        for m in mapped:
            if m not in seen:
                seen.add(m)
                dedup.append(m)
        return dedup
    except Exception as e:
        print("Error computing top features:", e)
        return []

# ============== CHURN PREDICTION ROUTES ==============

@app.route('/api/predict-churn', methods=['POST'])
def predict_churn():
    """Handle churn prediction for uploaded CSV file"""
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        if MODEL is None or PREPROCESSOR is None or SELECTOR is None:
            return jsonify({'success': False, 'message': 'Prediction model not loaded'}), 500
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': 'Invalid file type. Please upload a CSV file'}), 400
        
        filename = secure_filename(file.filename)
        job_id = str(uuid.uuid4())
        filepath = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
        file.save(filepath)
        
        df = pd.read_csv(filepath)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        user_id = session.get('user_id')
        cursor.execute(
            """INSERT INTO prediction_jobs 
               (userID, filename, status, total_records) 
               VALUES (%s, %s, 'processing', %s)""",
            (user_id, filename, len(df))
        )
        conn.commit()
        job_db_id = cursor.lastrowid
        
        expected_columns = [
            'RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 
            'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ]
        
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            cursor.execute(
                "UPDATE prediction_jobs SET status='failed', error_message=%s WHERE jobID=%s",
                (f"Missing columns: {', '.join(missing_cols)}", job_db_id)
            )
            conn.commit()
            cursor.close()
            conn.close()
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'success': False, 
                'message': f'Missing required columns: {", ".join(missing_cols)}'
            }), 400
        
        X = df[expected_columns].copy()
        
        X_processed = PREPROCESSOR.transform(X)
        X_selected = SELECTOR.transform(X_processed)
        
        predictions = MODEL.predict(X_selected)
        probabilities = MODEL.predict_proba(X_selected)[:, 1]
        
        shap_values = None
        selected_feature_names = []
        try:
            if EXPLAINER is not None:
                shap_values = EXPLAINER.shap_values(X_selected)
            if SELECTED_FEATURES and len(SELECTED_FEATURES) == X_selected.shape[1]:
                selected_feature_names = SELECTED_FEATURES
            else:
                proc_names, sel_names = _get_processed_and_selected_feature_names(PREPROCESSOR, SELECTOR)
                selected_feature_names = sel_names
        except Exception as e:
            print("Warning: Could not compute SHAP values or selected feature names:", e)
            shap_values = None
        
        high_risk = sum(1 for p in probabilities if p >= 0.7)
        medium_risk = sum(1 for p in probabilities if 0.4 <= p < 0.7)
        low_risk = sum(1 for p in probabilities if p < 0.4)
        avg_score = np.mean(probabilities)
        
        cursor.execute(
            """UPDATE prediction_jobs 
               SET status='completed', 
                   processed_records=%s,
                   high_risk_count=%s,
                   medium_risk_count=%s,
                   low_risk_count=%s,
                   average_churn_score=%s,
                   completedAt=NOW()
               WHERE jobID=%s""",
            (len(df), high_risk, medium_risk, low_risk, float(avg_score), job_db_id)
        )
        
        for idx, row in df.iterrows():
            customer_id = row.get('CustomerId', row.get('CustomerID', f'CUST_{idx}'))
            surname = row.get('Surname', None)
            probability = float(probabilities[idx])
            prediction = bool(predictions[idx])
            risk_level = calculate_risk_level(probability)
            
            top_features = None
            try:
                if shap_values is not None:
                    top_features = _get_top_features_for_selected_row(shap_values[idx], selected_feature_names, top_n=3)
            except Exception as e:
                print("Warning computing top features for row:", e)
                top_features = None
            
            cursor.execute(
                """INSERT INTO churn_predictions 
                   (jobID, customerID, surname, credit_score, geography, gender, 
                    age, tenure, balance, num_of_products, has_credit_card, 
                    is_active_member, estimated_salary, churn_probability, 
                    churn_prediction, risk_level)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    job_db_id, str(customer_id), surname,
                    int(row.get('CreditScore', 0)),
                    row.get('Geography', ''),
                    row.get('Gender', ''),
                    int(row.get('Age', 0)),
                    int(row.get('Tenure', 0)),
                    float(row.get('Balance', 0)),
                    int(row.get('NumOfProducts', 0)),
                    bool(row.get('HasCrCard', 0)),
                    bool(row.get('IsActiveMember', 0)),
                    float(row.get('EstimatedSalary', 0)),
                    probability,
                    prediction,
                    risk_level
                )
            )
        # Save feature importance if available
        if hasattr(MODEL, 'get_feature_importance'):
            try:
                importances = MODEL.get_feature_importance()
                if hasattr(SELECTOR, 'get_support'):
                    proc_names, sel_names = _get_processed_and_selected_feature_names(PREPROCESSOR, SELECTOR)
                    feature_names = sel_names if len(importances) == len(sel_names) else [f'feature_{i}' for i in range(len(importances))]
                else:
                    feature_names = [f'feature_{i}' for i in range(len(importances))]
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(20)
                
                for rank, (_, rowf) in enumerate(importance_df.iterrows(), 1):
                    cursor.execute(
                        """INSERT INTO feature_importance 
                           (jobID, feature_name, importance_score, rank)
                           VALUES (%s, %s, %s, %s)""",
                        (job_db_id, rowf['feature'], float(rowf['importance']), rank)
                    )
            except Exception as e:
                print(f"Warning: Could not save feature importance: {e}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        if os.path.exists(filepath):
            os.remove(filepath)
        
        try:
            log_system_event("/api/predict-churn", "POST", f"Prediction job {job_db_id} completed by user {user_id}", 200, user_id=user_id)
        except:
            pass
        
        return jsonify({
            'success': True,
            'message': 'Prediction completed successfully',
            'job_id': job_db_id,
            'statistics': {
                'total': len(df),
                'high_risk': high_risk,
                'medium_risk': medium_risk,
                'low_risk': low_risk,
                'avg_score': float(avg_score)
            }
        }), 200
        
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        try:
            if 'job_db_id' in locals() and 'cursor' in locals():
                cursor.execute(
                    "UPDATE prediction_jobs SET status='failed', error_message=%s WHERE jobID=%s",
                    (str(e), job_db_id)
                )
                conn.commit()
        except:
            pass
        try:
            user_id = session.get('user_id')
            log_system_event("/api/predict-churn", "POST", f"Prediction job failed: {e}", 500, user_id=user_id)
        except:
            pass
        return jsonify({
            'success': False,
            'message': 'An error occurred during prediction'
        }), 500

@app.route('/api/prediction-results/<int:job_id>', methods=['GET'])
def get_prediction_results(job_id):
    """Get prediction results for a specific job — access controlled by role and prediction_requests."""
    try:
        if not session.get('authenticated'):
            print("Unauthorized access attempt — no session")
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        user_id = session.get('user_id')
        role = session.get('role')

        print(f"Fetching job {job_id} for user {user_id} ({role})")

        # ---- Access Control Logic ---- #
        if role == 'IT admin':
            # IT admins can view any job that exists in prediction_requests
            cursor.execute("""
                SELECT r.*, j.*
                FROM prediction_requests r
                JOIN prediction_jobs j ON r.jobID = j.jobID
                WHERE r.jobID = %s
            """, (job_id,))
        elif role == 'Retail admin':
            # Retail admins can only view their own resolved jobs
            cursor.execute("""
                SELECT r.*, j.*
                FROM prediction_requests r
                JOIN prediction_jobs j ON r.jobID = j.jobID
                WHERE r.jobID = %s
                  AND r.requester_userID = %s
                  AND r.status = 'resolved'
            """, (job_id, user_id))
        else:
            cursor.close()
            conn.close()
            print("Unauthorized role tried accessing prediction results")
            return jsonify({'success': False, 'message': 'Unauthorized role'}), 403

        job = cursor.fetchone()
        if not job:
            cursor.close()
            conn.close()
            msg = 'Job not found or not yet resolved' if role == 'Retail admin' else 'Job not found'
            print(f"{msg}")
            return jsonify({'success': False, 'message': msg}), 404

        print(f"Job found: {job['jobID']} ({job.get('status')})")

        # ---- Fetch Predictions ---- #
        cursor.execute(
            """SELECT 
                customerID as customer_id,
                surname,
                churn_probability,
                churn_prediction,
                risk_level,
                credit_score,
                age,
                balance
               FROM churn_predictions 
               WHERE jobID=%s 
               ORDER BY churn_probability DESC""",
            (job_id,)
        )
        predictions = cursor.fetchall()
        print(f"Retrieved {len(predictions)} prediction records for job {job_id}")

        # ---- Fetch Feature Importance ---- #
        cursor.execute(
            """SELECT feature_name, importance_score 
               FROM feature_importance 
               WHERE jobID=%s 
               ORDER BY rank ASC 
               LIMIT 10""",
            (job_id,)
        )
        features = cursor.fetchall()

        # ---- SHAP Explanations ---- #
        try:
            if MODEL is not None and PREPROCESSOR is not None and SELECTOR is not None and EXPLAINER is not None and len(predictions) > 0:
                df_rows = pd.DataFrame(predictions)
                column_mapping = {
                    'credit_score': ['CreditScore', 'credit_score', 'creditScore'],
                    'geography': ['Geography', 'geography'],
                    'gender': ['Gender', 'gender'],
                    'age': ['Age', 'age'],
                    'tenure': ['Tenure', 'tenure'],
                    'balance': ['Balance', 'balance'],
                    'num_of_products': ['NumOfProducts', 'num_of_products', 'numOfProducts'],
                    'has_credit_card': ['HasCrCard', 'has_credit_card', 'hasCreditCard'],
                    'is_active_member': ['IsActiveMember', 'is_active_member', 'isActiveMember'],
                    'estimated_salary': ['EstimatedSalary', 'estimated_salary', 'estimatedSalary']
                }
                try:
                    proc_info = PREPROCESSOR.transformers_
                    orig_num_cols, orig_cat_cols = [], []
                    for name, transformer, cols in PREPROCESSOR.transformers_:
                        if name == 'num':
                            orig_num_cols = list(cols)
                        elif name == 'cat':
                            orig_cat_cols = list(cols)
                except Exception:
                    orig_num_cols, orig_cat_cols = [], []

                X_raw = pd.DataFrame()
                for col in orig_num_cols + orig_cat_cols:
                    found = None
                    lower_col = col.lower()
                    for src_col in df_rows.columns:
                        if src_col.lower() == col.lower():
                            found = df_rows[src_col]
                            break
                    if found is None:
                        for key, variants in column_mapping.items():
                            if col.lower() in [v.lower() for v in variants]:
                                for variant in variants:
                                    if variant in df_rows.columns:
                                        found = df_rows[variant]
                                        break
                                if found is not None:
                                    break
                    X_raw[col] = found.values if found is not None else np.nan

                if X_raw.shape[1] == 0:
                    X_raw = df_rows.copy()

                X_proc = PREPROCESSOR.transform(X_raw)
                X_sel = SELECTOR.transform(X_proc)
                shap_vals = EXPLAINER.shap_values(X_sel)

                if SELECTED_FEATURES and len(SELECTED_FEATURES) == X_sel.shape[1]:
                    sel_feature_names = SELECTED_FEATURES
                else:
                    _, sel_feature_names = _get_processed_and_selected_feature_names(PREPROCESSOR, SELECTOR)

                for i, rec in enumerate(predictions):
                    try:
                        row_shap = shap_vals[i]
                        top_feats = _get_top_features_for_selected_row(row_shap, sel_feature_names, top_n=3)
                        predictions[i]['top_features'] = top_feats
                    except Exception as e:
                        print("SHAP computation failed for row:", i, e)
                        predictions[i]['top_features'] = []
            else:
                print("Skipping SHAP computation — model components missing or no predictions")
                for i in range(len(predictions)):
                    predictions[i]['top_features'] = []
        except Exception as e:
            print("Error computing SHAP values in get_prediction_results:", e)
            traceback.print_exc()
            for i in range(len(predictions)):
                predictions[i]['top_features'] = []

        cursor.close()
        conn.close()

        # ---- Return Response ---- #
        return jsonify({
            'success': True,
            'job': {
                'id': job['jobID'],
                'description': job.get('description'),
                'status': job.get('status'),
                'filename': job.get('filename'),
                'requested_by': job.get('requester_username'),
                'resolved_by': job.get('resolvedByUsername'),
                'department': job.get('department'),
                'requested_at': job.get('requestedAt').isoformat() if job.get('requestedAt') else None,
                'resolved_at': job.get('resolvedAt').isoformat() if job.get('resolvedAt') else None
            },
            'statistics': {
                'total': job.get('total_records'),
                'high_risk': job.get('high_risk_count'),
                'medium_risk': job.get('medium_risk_count'),
                'low_risk': job.get('low_risk_count'),
                'avg_score': float(job.get('average_churn_score', 0.0))
            },
            'results': predictions,
            'top_features': [{'name': f['feature_name'], 'importance': float(f['importance_score'])} for f in features],
            'timestamp': job.get('resolvedAt').isoformat() if job.get('resolvedAt') else job.get('createdAt').isoformat()
        }), 200

    except Exception as e:
        print(f"Error fetching results: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Error fetching results'}), 500


@app.route('/api/predict', methods=['POST'])
def predict_json():
    """
    JSON-based prediction endpoint (real-time or batch).
    """
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401

        if MODEL is None or PREPROCESSOR is None or SELECTOR is None:
            return jsonify({'success': False, 'message': 'Prediction model not loaded'}), 500

        payload = request.get_json()
        if payload is None:
            return jsonify({'success': False, 'message': 'No JSON payload received'}), 400

        if isinstance(payload, dict):
            df_input = pd.DataFrame([payload])
        elif isinstance(payload, list):
            df_input = pd.DataFrame(payload)
        else:
            return jsonify({'success': False, 'message': 'Invalid payload format (expect JSON object or array)'}), 400

        X_proc = PREPROCESSOR.transform(df_input)
        X_sel = SELECTOR.transform(X_proc)

        probs = MODEL.predict_proba(X_sel)[:, 1]
        preds = (probs >= 0.5).astype(int)

        shap_vals = None
        try:
            if EXPLAINER is not None:
                shap_vals = EXPLAINER.shap_values(X_sel)
        except Exception as e:
            print("Warning: SHAP computation failed for predict route:", e)
            shap_vals = None

        if SELECTED_FEATURES and len(SELECTED_FEATURES) == X_sel.shape[1]:
            sel_feature_names = SELECTED_FEATURES
        else:
            _, sel_feature_names = _get_processed_and_selected_feature_names(PREPROCESSOR, SELECTOR)

        results = []
        for i in range(len(df_input)):
            prob = float(probs[i])
            pred = int(preds[i])
            risk = calculate_risk_level(prob)
            top_feats = []
            if shap_vals is not None:
                try:
                    top_feats = _get_top_features_for_selected_row(shap_vals[i], sel_feature_names, top_n=3)
                except Exception as e:
                    top_feats = []

            results.append({
                "customer_id": int(df_input.iloc[i].get("CustomerId", i)),
                "surname": str(df_input.iloc[i].get("Surname", "")),
                "churn_probability": prob,
                "churn_prediction": pred,
                "risk_level": risk,
                "top_features": top_feats
            })

        high = sum(1 for r in results if r['risk_level'] == 'high')
        med = sum(1 for r in results if r['risk_level'] == 'medium')
        low = sum(1 for r in results if r['risk_level'] == 'low')
        avg = float(np.mean([r['churn_probability'] for r in results])) if results else 0.0

        response = {
            "timestamp": datetime.utcnow().isoformat(),
            "statistics": {
                "total": len(results),
                "high_risk": high,
                "medium_risk": med,
                "low_risk": low,
                "avg_score": avg
            },
            "results": results
        }
        try:
            log_system_event("/api/predict", "POST", f"Real-time prediction by user {session.get('user_id')}", 200, user_id=session.get('user_id'))
        except:
            pass

        return jsonify(response), 200

    except Exception as e:
        print("Error in /api/predict:", e)
        traceback.print_exc()
        try:
            log_system_event("/api/predict", "POST", f"Prediction failed: {e}", 500, user_id=session.get('user_id'))
        except:
            pass
        return jsonify({'success': False, 'message': 'Prediction failed'}), 500

# ----------------------------
# Prediction Request Routes
# ----------------------------
@app.route('/api/submit-prediction-request', methods=['POST'])
def submit_prediction_request():
    """Retail admin submits a request for IT to run a prediction job."""
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401

        role = session.get('role')
        if role not in ('Retail admin', 'Retail Banking Admin', 'Business analyst'):
            # friendly message if role mismatch
            return jsonify({'success': False, 'message': 'Only retail/biz users can submit requests'}), 403

        data = request.json or {}
        description = data.get('description', '').strip()
        department = data.get('department', 'Retail Banking')

        if not description:
            return jsonify({'success': False, 'message': 'Description is required'}), 400

        requester_userID = session.get('user_id')
        requester_username = session.get('username')

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO prediction_requests 
               (requester_userID, requester_username, department, description, status)
               VALUES (%s, %s, %s, %s, 'pending')""",
            (requester_userID, requester_username, department, description)
        )
        conn.commit()
        request_id = cursor.lastrowid
        cursor.close()
        conn.close()

        try:
            log_system_event("/api/submit-prediction-request", "POST", f"Request {request_id} submitted by {requester_username}", 201, user_id=requester_userID)
        except Exception:
            pass

        print(f"Prediction request submitted: id={request_id}, by={requester_username}")
        return jsonify({'success': True, 'message': 'Request submitted', 'requestID': request_id}), 201

    except Exception as e:
        print(f"submit_prediction_request error: {e}")
        traceback.print_exc()
        try:
            log_system_event("/api/submit-prediction-request", "POST", f"Submission error: {e}", 500, user_id=session.get('user_id'))
        except:
            pass
        return jsonify({'success': False, 'message': 'Server error'}), 500


@app.route('/api/pending-requests', methods=['GET'])
def get_pending_requests():
    """IT admin: list all pending requests (or all requests, optionally filtered)."""
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401

        role = session.get('role')
        if role != 'IT admin':
            return jsonify({'success': False, 'message': 'Access denied'}), 403

        # optional query param ?status=all|pending|resolved
        status = request.args.get('status', 'pending')
        limit = int(request.args.get('limit', 200))

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        if status == 'all':
            cursor.execute("""
                SELECT r.requestID, r.requester_userID, r.requester_username, r.department, r.description,
                       r.status, r.requestedAt, r.resolvedAt, r.resolvedBy, r.resolvedByUsername, r.jobID
                FROM prediction_requests r
                ORDER BY r.requestedAt DESC
                LIMIT %s
            """, (limit,))
        else:
            cursor.execute("""
                SELECT r.requestID, r.requester_userID, r.requester_username, r.department, r.description,
                       r.status, r.requestedAt, r.resolvedAt, r.resolvedBy, r.resolvedByUsername, r.jobID
                FROM prediction_requests r
                WHERE r.status = %s
                ORDER BY r.requestedAt DESC
                LIMIT %s
            """, (status, limit))

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Convert datetime objects to iso strings if needed
        for r in rows:
            if isinstance(r.get('requestedAt'), datetime):
                r['requestedAt'] = r['requestedAt'].isoformat()
            if isinstance(r.get('resolvedAt'), datetime):
                r['resolvedAt'] = r['resolvedAt'].isoformat()

        return jsonify({'success': True, 'requests': rows}), 200

    except Exception as e:
        print(f"get_pending_requests error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error'}), 500


@app.route('/api/resolve-request', methods=['POST'])
def resolve_request():
    """
    IT admin marks a request resolved. Body: { requestID: int, jobID: int }
    This updates status -> 'resolved' and sets jobID, resolvedAt, resolvedBy.
    """
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401

        role = session.get('role')
        if role != 'IT admin':
            return jsonify({'success': False, 'message': 'Access denied'}), 403

        data = request.json or {}
        request_id = data.get('requestID')
        job_id = data.get('jobID')

        if not request_id or not job_id:
            return jsonify({'success': False, 'message': 'requestID and jobID are required'}), 400

        resolved_by = session.get('user_id')
        resolved_by_username = session.get('username')

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check request exists and is pending
        cursor.execute("SELECT status FROM prediction_requests WHERE requestID = %s", (request_id,))
        row = cursor.fetchone()
        if not row:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Request not found'}), 404

        existing_status = row[0]
        if existing_status == 'resolved':
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Request already resolved'}), 409

        cursor.execute("""
            UPDATE prediction_requests
            SET status='resolved',
                jobID=%s,
                resolvedAt=NOW(),
                resolvedBy=%s,
                resolvedByUsername=%s
            WHERE requestID=%s
        """, (job_id, resolved_by, resolved_by_username, request_id))
        conn.commit()
        affected = cursor.rowcount
        cursor.close()
        conn.close()

        if affected == 0:
            return jsonify({'success': False, 'message': 'No changes made'}), 400

        try:
            log_system_event("/api/resolve-request", "POST", f"Request {request_id} resolved with jobID {job_id} by {resolved_by_username}", 200, user_id=resolved_by)
        except:
            pass

        print(f"Request {request_id} resolved by {resolved_by_username} -> job {job_id}")
        return jsonify({'success': True, 'message': 'Request resolved'}), 200

    except Exception as e:
        print(f"resolve_request error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error'}), 500


@app.route('/api/my-requests', methods=['GET'])
def my_requests():
    """Retail admin: list requests submitted by the current user."""
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401

        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'message': 'Invalid session'}), 401

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT requestID, department, description, status, requestedAt, resolvedAt, jobID, resolvedByUsername
            FROM prediction_requests
            WHERE requester_userID = %s
            ORDER BY requestedAt DESC
            LIMIT 200
        """, (user_id,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # normalize times
        for r in rows:
            if isinstance(r.get('requestedAt'), datetime):
                r['requestedAt'] = r['requestedAt'].isoformat()
            if isinstance(r.get('resolvedAt'), datetime):
                r['resolvedAt'] = r['resolvedAt'].isoformat()

        return jsonify({'success': True, 'requests': rows}), 200

    except Exception as e:
        print(f"my_requests error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error'}), 500

@app.route('/api/prediction-requests', methods=['POST'])
def create_prediction_request():
    """Retail admin (or any authenticated user) creates a prediction request"""
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401

        data = request.json or {}
        description = data.get('description', '').strip()
        department = data.get('department', 'Retail Banking').strip()
        notify_email = data.get('notifyEmail') or data.get('notify_email') or None

        requester_id = session.get('user_id')
        requester_username = session.get('username') or 'unknown'

        if not description:
            return jsonify({'success': False, 'message': 'Description is required'}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prediction_requests
              (requester_userID, requester_username, department, description)
            VALUES (%s, %s, %s, %s)
        """, (requester_id, requester_username, department, description))
        conn.commit()
        request_id = cursor.lastrowid

        # Optionally save notify email in a lightweight way (could be separate column)
        if notify_email:
            try:
                cursor.execute("UPDATE prediction_requests SET resolvedByUsername = %s WHERE requestID = %s", (notify_email, request_id))
                conn.commit()
            except Exception as e:
                print("Warning: notify email save failed:", e)

        cursor.close()
        conn.close()

        try:
            log_system_event('/api/prediction-requests', 'POST', f'New prediction request {request_id} created by {requester_username}', 201, user_id=requester_id)
        except:
            pass

        print(f"Created prediction request {request_id} by user {requester_username}")
        return jsonify({'success': True, 'message': 'Request created', 'requestID': request_id}), 201
    except Exception as e:
        print(f"Error creating prediction request: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error'}), 500


@app.route('/api/prediction-requests', methods=['GET'])
def list_prediction_requests():
    """
    List prediction requests.
    Query params:
      - status=all|pending|resolved
      - mine=1   -> only requests created by current user (Retail admin view)
    IT admins will get all requests if mine not set.
    """
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401

        user_id = session.get('user_id')
        role = session.get('role')

        status = request.args.get('status', 'all')
        mine = request.args.get('mine', None)

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        base_q = "SELECT requestID, requester_userID, requester_username, department, description, status, requestedAt, resolvedAt, resolvedBy, resolvedByUsername, jobID FROM prediction_requests"
        conditions = []
        params = []

        if status in ('pending', 'resolved'):
            conditions.append("status = %s")
            params.append(status)

        # If mine param present and truthy -> restrict to current user
        if mine and mine in ('1', 'true', 'True'):
            conditions.append("requester_userID = %s")
            params.append(user_id)
        else:
            # If user is not IT admin and mine not requested, restrict to user's requests as default
            if role != 'IT admin':
                conditions.append("requester_userID = %s")
                params.append(user_id)

        if conditions:
            base_q += " WHERE " + " AND ".join(conditions)

        base_q += " ORDER BY requestedAt DESC LIMIT 500"

        cursor.execute(base_q, tuple(params))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Mark whether current session user can resolve (IT admin)
        for r in rows:
            r['_canResolve'] = True if session.get('role') == 'IT admin' else False
            # Normalize timestamps for frontend
            if isinstance(r.get('requestedAt'), (datetime, )):
                r['requestedAt'] = r['requestedAt'].isoformat()
            if isinstance(r.get('resolvedAt'), (datetime, )):
                r['resolvedAt'] = r['resolvedAt'].isoformat()

        return jsonify({'success': True, 'requests': rows}), 200
    except Exception as e:
        print(f"Error listing prediction requests: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error'}), 500


@app.route('/api/prediction-requests/<int:request_id>/resolve', methods=['POST'])
def resolve_prediction_request(request_id):
    """
    Mark a prediction request as resolved by an IT admin.
    Body: { jobID: <int> }
    Server-side validation: jobID must exist in prediction_jobs.
    Sends email to requester if email available (we saved into resolvedByUsername earlier as notify email fallback).
    """
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401

        if session.get('role') != 'IT admin':
            return jsonify({'success': False, 'message': 'Access denied'}), 403

        data = request.json or {}
        job_id = data.get('jobID')
        if not job_id:
            return jsonify({'success': False, 'message': 'jobID is required'}), 400

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Validate request exists
        cursor.execute("SELECT * FROM prediction_requests WHERE requestID = %s", (request_id,))
        req = cursor.fetchone()
        if not req:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Request not found'}), 404

        # Validate job exists
        cursor.execute("SELECT jobID, filename FROM prediction_jobs WHERE jobID = %s", (job_id,))
        job = cursor.fetchone()
        if not job:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Provided jobID does not exist'}), 400

        resolved_by = session.get('user_id')
        resolved_by_username = session.get('username')

        cursor.execute("""
            UPDATE prediction_requests
            SET status='resolved', resolvedAt = NOW(), resolvedBy = %s, resolvedByUsername = %s, jobID = %s
            WHERE requestID = %s
        """, (resolved_by, resolved_by_username, job_id, request_id))
        conn.commit()

        # Fetch requester info to email
        cursor.execute("SELECT u.email FROM users u WHERE u.userID = %s", (req['requester_userID'],))
        requester_row = cursor.fetchone()
        requester_email = requester_row.get('email') if requester_row else None

        cursor.close()
        conn.close()

        try:
            log_system_event(f"/api/prediction-requests/{request_id}/resolve", "POST", f"Request {request_id} resolved -> job {job_id}", 200, user_id=resolved_by)
        except:
            pass

        # Send email to requester if email exists (use send_otp_email helper but with different subject/body)
        try:
            if requester_email:
                subject = "Your prediction request has been resolved"
                body_html = f"""
                    <html><body>
                      <h3>Prediction Request Resolved</h3>
                      <p>Hi {req.get('requester_username')},</p>
                      <p>Your prediction request (ID: {request_id}) has been completed by {resolved_by_username}.</p>
                      <p>Job ID: <strong>{job_id}</strong> — file: {job.get('filename')}</p>
                      <p>You can view the results at: <a href="#">Prediction Results</a></p>
                      <br><p>Regards,<br/>LoyaltyLens Team</p>
                    </body></html>
                """
                try:
                    # re-use SMTP helper but send_otp_email expects otp; we'll craft quick mail send
                    server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
                    server.starttls()
                    server.login(EMAIL_CONFIG['email'], EMAIL_CONFIG['password'])
                    msg = MIMEMultipart()
                    msg['From'] = EMAIL_CONFIG['email']
                    msg['To'] = requester_email
                    msg['Subject'] = subject
                    msg.attach(MIMEText(body_html, 'html'))
                    server.send_message(msg)
                    server.quit()
                    print(f"Notification email sent to {requester_email} for request {request_id}")
                except Exception as e:
                    print("Warning: failed to send resolution email:", e)
        except Exception as e:
            print("Email notify fallback failed:", e)

        return jsonify({'success': True, 'message': 'Request marked resolved', 'jobID': job_id}), 200

    except Exception as e:
        print(f"Error resolving request {request_id}: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error'}), 500


@app.route('/api/retail-dashboard-stats', methods=['GET'])
def retail_dashboard_stats():
    """Return summary stats for retail dashboard (pending, resolved, totals, avg resolution time)"""
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401

        user_id = session.get('user_id')
        role = session.get('role')

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # If IT admin, summary over all requests, else only user requests
        if role == 'IT admin':
            cursor.execute("SELECT COUNT(*) AS cnt FROM prediction_requests")
            total = cursor.fetchone()['cnt'] or 0
            cursor.execute("SELECT COUNT(*) AS cnt FROM prediction_requests WHERE status='pending'")
            pending = cursor.fetchone()['cnt'] or 0
            cursor.execute("SELECT COUNT(*) AS cnt FROM prediction_requests WHERE status='resolved'")
            resolved = cursor.fetchone()['cnt'] or 0
            cursor.execute("""
                SELECT AVG(TIMESTAMPDIFF(SECOND, requestedAt, resolvedAt))/3600 AS avg_hours
                FROM prediction_requests
                WHERE status = 'resolved' AND resolvedAt IS NOT NULL
            """)
            avg_row = cursor.fetchone()
        else:
            cursor.execute("SELECT COUNT(*) AS cnt FROM prediction_requests WHERE requester_userID=%s", (user_id,))
            total = cursor.fetchone()['cnt'] or 0
            cursor.execute("SELECT COUNT(*) AS cnt FROM prediction_requests WHERE requester_userID=%s AND status='pending'", (user_id,))
            pending = cursor.fetchone()['cnt'] or 0
            cursor.execute("SELECT COUNT(*) AS cnt FROM prediction_requests WHERE requester_userID=%s AND status='resolved'", (user_id,))
            resolved = cursor.fetchone()['cnt'] or 0
            cursor.execute("""
                SELECT AVG(TIMESTAMPDIFF(SECOND, requestedAt, resolvedAt))/3600 AS avg_hours
                FROM prediction_requests
                WHERE status = 'resolved' AND resolvedAt IS NOT NULL AND requester_userID=%s
            """, (user_id,))
            avg_row = cursor.fetchone()

        cursor.close()
        conn.close()

        avg_hours = avg_row.get('avg_hours') if avg_row else None
        avg_text = f"{round(avg_hours,2)}h" if avg_hours else "-"

        return jsonify({'success': True, 'data': {
            'total': int(total),
            'pending': int(pending),
            'resolved': int(resolved),
            'avg_resolution': avg_text
        }}), 200
    except Exception as e:
        print("Error in retail_dashboard_stats:", e)
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error'}), 500

# ----------------------------
# USER MANAGEMENT ROUTES
# ----------------------------

@app.route('/api/register-user', methods=['POST'])
def register_user():
    """Register a new user in the LoyaltyLens system"""
    try:
        data = request.json
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        role = data.get('role')

        if not username or not email or not password or not role:
            return jsonify({'success': False, 'message': 'All fields are required'}), 400

        if len(password) < 8:
            return jsonify({'success': False, 'message': 'Password must be at least 8 characters long'}), 400

        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT userID FROM users WHERE username=%s OR email=%s", (username, email))
        existing_user = cursor.fetchone()
        if existing_user:
            cursor.close()
            conn.close()
            try:
                log_system_event("/api/register-user", "POST", f"Registration failed - duplicate {username} or {email}", 409, user_id=session.get('user_id'))
            except:
                pass
            return jsonify({'success': False, 'message': 'Username or email already exists'}), 409

        cursor.execute(
            """INSERT INTO users (username, email, password_hash, role)
               VALUES (%s, %s, %s, %s)""",
            (username, email, password_hash, role)
        )
        conn.commit()
        cursor.close()
        conn.close()

        try:
            log_system_event("/api/register-user", "POST", f"User registered: {username}", 201, user_id=session.get('user_id'))
        except:
            pass

        return jsonify({
            'success': True,
            'message': f'User \"{username}\" registered successfully!'
        }), 201

    except Exception as e:
        print(f"Registration error: {e}")
        traceback.print_exc()
        try:
            log_system_event("/api/register-user", "POST", f"Registration error: {e}", 500, user_id=session.get('user_id'))
        except:
            pass
        return jsonify({'success': False, 'message': 'Server error'}), 500

@app.route('/api/lookup-user', methods=['POST'])
def lookup_user():
    try:
        data = request.json
        username = data.get('username')

        if not username:
            return jsonify({'success': False, 'message': 'Username is required'}), 400

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT username, email, role FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if not user:
            return jsonify({'success': False, 'message': 'User not found'}), 404

        return jsonify({'success': True, 'user': user}), 200

    except Exception as e:
        print(f"Lookup user error: {e}")
        return jsonify({'success': False, 'message': 'Server error'}), 500

@app.route('/api/update-user', methods=['PUT'])
def update_user():
    try:
        data = request.json
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        role = data.get('role')

        if not username or not email or not role:
            return jsonify({'success': False, 'message': 'Missing required fields'}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        if password:
            import bcrypt
            hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            cursor.execute("""
                UPDATE users 
                SET email=%s, password_hash=%s, role=%s, updatedAt=NOW() 
                WHERE username=%s
            """, (email, hashed, role, username))
        else:
            cursor.execute("""
                UPDATE users 
                SET email=%s, role=%s, updatedAt=NOW() 
                WHERE username=%s
            """, (email, role, username))

        affected = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()

        if affected == 0:
            return jsonify({'success': False, 'message': 'User not found or no changes made'}), 404

        return jsonify({'success': True, 'message': 'User updated successfully'}), 200

    except Exception as e:
        print(f"Update user error: {e}")
        return jsonify({'success': False, 'message': 'Server error'}), 500

@app.route('/api/revoke-user', methods=['DELETE'])
def revoke_user():
    try:
        data = request.json
        username = data.get('username')

        if not username:
            return jsonify({'success': False, 'message': 'Username is required'}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        # Soft revoke (avoid deletion to keep FK integrity)
        cursor.execute("""
            UPDATE users 
            SET role = 'Revoked', is_active = FALSE, updatedAt = NOW() 
            WHERE username = %s
        """, (username,))

        conn.commit()
        affected = cursor.rowcount
        cursor.close()
        conn.close()

        if affected == 0:
            return jsonify({'success': False, 'message': 'User not found'}), 404

        return jsonify({'success': True, 'message': 'User access revoked successfully'}), 200

    except Exception as e:
        print(f"Revoke user error: {e}")
        return jsonify({'success': False, 'message': 'Server error'}), 500

@app.route('/api/prediction-history', methods=['GET'])
def get_prediction_history():
    """Get user's prediction history"""
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        user_id = session.get('user_id')
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(
            """SELECT 
                jobID,
                filename,
                status,
                total_records,
                high_risk_count,
                medium_risk_count,
                low_risk_count,
                average_churn_score,
                createdAt,
                completedAt
               FROM prediction_jobs 
               WHERE userID=%s 
               ORDER BY createdAt DESC
               LIMIT 50""",
            (user_id,)
        )
        history = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        try:
            log_system_event("/api/prediction-history", "GET", f"User {user_id} fetched prediction history", 200, user_id=user_id)
        except:
            pass
        
        return jsonify({
            'success': True,
            'history': history
        }), 200
        
    except Exception as e:
        print(f"Error fetching history: {e}")
        traceback.print_exc()
        try:
            log_system_event("/api/prediction-history", "GET", f"Error fetching history: {e}", 500, user_id=session.get('user_id'))
        except:
            pass
        return jsonify({'success': False, 'message': 'Error fetching history'}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout user"""
    try:
        user_id = session.get('user_id')
        session.clear()
        try:
            log_system_event("/api/logout", "POST", f"User {user_id} logged out", 200, user_id=user_id)
        except:
            pass
        return jsonify({'success': True, 'message': 'Logged out successfully'}), 200
    except Exception as e:
        print("Logout error:", e)
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Logout failed'}), 500

@app.route('/api/check-auth', methods=['GET'])
def check_auth():
    """Check if user is authenticated"""
    if session.get('authenticated'):
        return jsonify({
            'authenticated': True,
            'username': session.get('username'),
            'role': session.get('role')
        }), 200
    return jsonify({'authenticated': False}), 401

@app.route('/api/admin-stats', methods=['GET'])
def get_admin_stats():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT COUNT(*) AS total_users FROM users")
        total_users = cursor.fetchone()['total_users']

        try:
            cursor.execute("SELECT COUNT(*) AS active_users FROM users WHERE is_active = TRUE")
            active_users = cursor.fetchone()['active_users']
        except:
            cursor.execute("SELECT COUNT(*) AS active_users FROM users WHERE role != 'Revoked'")
            active_users = cursor.fetchone()['active_users']

        try:
            cursor.execute("SELECT COUNT(*) AS revoked_users FROM users WHERE is_active = FALSE")
            revoked_users = cursor.fetchone()['revoked_users']
        except:
            cursor.execute("SELECT COUNT(*) AS revoked_users FROM users WHERE role = 'Revoked'")
            revoked_users = cursor.fetchone()['revoked_users']

        cursor.execute("SELECT COUNT(*) AS total_logs FROM system_logs")
        total_logs = cursor.fetchone()['total_logs']

        cursor.close()
        conn.close()

        return jsonify({'success': True, 'data': {
            'total_users': total_users,
            'active_users': active_users,
            'revoked_users': revoked_users,
            'total_logs': total_logs
        }}), 200
    except Exception as e:
        print(f"Error fetching admin stats: {e}")
        return jsonify({'success': False, 'message': 'Error fetching admin stats'}), 500

@app.route('/api/recent-logs', methods=['GET'])
def get_recent_logs():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT s.timestamp, u.username, s.action, s.status
            FROM system_logs s
            LEFT JOIN users u ON s.userID = u.userID
            ORDER BY s.timestamp DESC
            LIMIT 10
        """)
        logs = cursor.fetchall()
        cursor.close()
        conn.close()
        for r in logs:
            if isinstance(r.get('timestamp'), (datetime, )):
                r['timestamp'] = r['timestamp'].isoformat()
        return jsonify({'success': True, 'logs': logs}), 200
    except Exception as e:
        print(f"Error fetching recent logs: {e}")
        return jsonify({'success': False, 'message': 'Error fetching logs'}), 500

@app.route('/api/system-health', methods=['GET'])
def api_system_health():
    try:
        db_ok, db_err = False, None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            conn.close()
            db_ok = True
        except Exception as e:
            db_ok = False
            db_err = str(e)

        model_info = get_model_artifacts_info()

        uptime_seconds = (datetime.utcnow() - APP_START_TIME).total_seconds()
        uptime_iso = APP_START_TIME.isoformat() + 'Z'

        api_calls_24h = 0
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT COUNT(*) AS cnt
                FROM api_metrics
                WHERE timestamp >= NOW() - INTERVAL 1 DAY
            """)
            r = cursor.fetchone()
            api_calls_24h = int(r['cnt']) if r and r.get('cnt') is not None else 0
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"count api_metrics error: {e}")
            api_calls_24h = 0

        resp = {'success': True, 'data': {
            'db': {'status': 'active' if db_ok else 'down', 'error': db_err},
            'model': model_info,
            'api': {
                'app_start_time': uptime_iso,
                'uptime_seconds': int(uptime_seconds),
                'api_calls_last_24h': int(api_calls_24h)
            }
        }}
        return jsonify(resp), 200
    except Exception as e:
        print(f"api_system_health error: {e}")
        return jsonify({'success': False, 'message': 'Error fetching system health'}), 500

@app.route('/api/model-performance', methods=['GET'])
def api_model_performance():
    try:
        m = get_model_artifacts_info()
        metrics = m.get('metrics') or {}
        acc = metrics.get('accuracy') or metrics.get('acc') or metrics.get('roc_auc') or None
        precision = metrics.get('precision') or metrics.get('prec') or None
        recall = metrics.get('recall') or metrics.get('rec') or None

        return jsonify({'success': True, 'data': {
            'model_path': m.get('model_path'),
            'file_exists': m.get('file_exists'),
            'file_mtime_iso': m.get('file_mtime_iso'),
            'metrics': {'accuracy': acc, 'precision': precision, 'recall': recall}
        }}), 200
    except Exception as e:
        print(f"api_model_performance error: {e}")
        return jsonify({'success': False, 'message': 'Error fetching model performance'}), 500

@app.route('/api/api-metrics', methods=['GET'])
def api_metrics():
    """Return quick API metrics: requests last hour etc."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN timestamp >= NOW() - INTERVAL 1 HOUR THEN 1 ELSE 0 END) AS requests_last_hour,
                SUM(CASE WHEN timestamp >= NOW() - INTERVAL 24 HOUR THEN 1 ELSE 0 END) AS requests_last_24h
            FROM api_metrics
        """)
        data = cursor.fetchone() or {}
        cursor.close()
        conn.close()
        return jsonify({'success': True, 'data': {
            'requests_last_hour': int(data.get('requests_last_hour') or 0),
            'requests_last_24h': int(data.get('requests_last_24h') or 0)
        }}), 200
    except Exception as e:
        print(f"api_metrics error: {e}")
        return jsonify({'success': False, 'message': 'Error fetching api metrics'}), 500

@app.route('/api/system-logs', methods=['GET'])
def get_system_logs():
    """Retrieve recent system logs (IT admin only)"""
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        role = session.get('role')
        if role != 'IT admin':
            return jsonify({'success': False, 'message': 'Access denied'}), 403

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                l.logID, 
                u.username, 
                l.endpoint, 
                l.method, 
                l.action_description, 
                l.status_code, 
                l.timestamp
            FROM system_logs l
            LEFT JOIN users u ON l.userID = u.userID
            ORDER BY l.timestamp DESC
            LIMIT 500
        """)
        logs = cursor.fetchall()
        cursor.close()
        conn.close()

        return jsonify({'success': True, 'logs': logs}), 200
    except Exception as e:
        print("Error fetching logs:", e)
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error'}), 500

# Optional: serve dashboard HTML if you place the html in project root
@app.route('/dashboard')
def dashboard():
    try:
        return send_from_directory('.', 'prediction_table.html')
    except Exception as e:
        print("Error serving dashboard:", e)
        return jsonify({'success': False, 'message': 'Dashboard not found'}), 404

if __name__ == '__main__':
    print("="*50)
    print("Starting Flask Application")
    print("="*50)
    if MODEL is not None:
        print("✅ Churn prediction model loaded")
    else:
        print("⚠️  Churn prediction model not loaded")
    print("="*50)
    # debug True ok for local dev - set to False in production
    app.run(debug=True, port=5000)
