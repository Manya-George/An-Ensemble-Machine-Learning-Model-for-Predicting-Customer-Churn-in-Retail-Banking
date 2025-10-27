from flask import Flask, request, jsonify, session
from flask_cors import CORS
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

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
CORS(app, supports_credentials=True)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MODEL_PATH = 'catboost_churn_model.joblib'

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database connection
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

# Load the trained model
try:
    model_artifacts = joblib.load(MODEL_PATH)
    MODEL = model_artifacts['model']
    PREPROCESSOR = model_artifacts['preprocessor']
    SELECTOR = model_artifacts['selector']
    print("✅ Churn prediction model loaded successfully!")
except Exception as e:
    print(f"⚠️  Warning: Could not load model: {e}")
    MODEL = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

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

# ============== AUTHENTICATION ROUTES ==============

@app.route('/api/login', methods=['POST'])
def login():
    """Handle user login"""
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(
            "SELECT userID, username, email, password_hash, role FROM users WHERE username = %s",
            (username,)
        )
        user = cursor.fetchone()
        
        if not user:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
        
        if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
        
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
            return jsonify({'success': False, 'message': 'Failed to send OTP email'}), 500
        
        session['temp_user_id'] = user['userID']
        session['username'] = user['username']
        session['role'] = user['role']
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'OTP sent successfully',
            'masked_email': masked_email
        }), 200
        
    except Exception as e:
        print(f"Login error: {e}")
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
        
        session['user_id'] = user_id
        session['authenticated'] = True
        session.pop('temp_user_id', None)
        
        role = session.get('role')
        username = session.get('username')
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'OTP verified successfully',
            'role': role,
            'username': username
        }), 200
        
    except Exception as e:
        print(f"OTP verification error: {e}")
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
        
        return jsonify({
            'success': True,
            'message': 'OTP resent successfully'
        }), 200
        
    except Exception as e:
        print(f"Resend OTP error: {e}")
        return jsonify({'success': False, 'message': 'Server error'}), 500

# ============== CHURN PREDICTION ROUTES ==============

@app.route('/api/predict-churn', methods=['POST'])
def predict_churn():
    """Handle churn prediction for uploaded CSV file"""
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        if MODEL is None:
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
        
        # The preprocessor was trained on all columns except 'Exited'
        # We need to include all columns that were in the training data
        expected_columns = [
            'RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 
            'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ]
        
        # Check for missing columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            cursor.execute(
                "UPDATE prediction_jobs SET status='failed', error_message=%s WHERE jobID=%s",
                (f"Missing columns: {', '.join(missing_cols)}", job_db_id)
            )
            conn.commit()
            cursor.close()
            conn.close()
            return jsonify({
                'success': False, 
                'message': f'Missing required columns: {", ".join(missing_cols)}'
            }), 400
        
        # Use all columns (preprocessor will handle which ones to use)
        X = df[expected_columns].copy()
        
        X_processed = PREPROCESSOR.transform(X)
        X_selected = SELECTOR.transform(X_processed)
        
        predictions = MODEL.predict(X_selected)
        probabilities = MODEL.predict_proba(X_selected)[:, 1]
        
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
        
        if hasattr(MODEL, 'get_feature_importance'):
            try:
                importances = MODEL.get_feature_importance()
                feature_names = [f'feature_{i}' for i in range(len(importances))]
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(20)
                
                for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
                    cursor.execute(
                        """INSERT INTO feature_importance 
                           (jobID, feature_name, importance_score, rank)
                           VALUES (%s, %s, %s, %s)""",
                        (job_db_id, row['feature'], float(row['importance']), rank)
                    )
            except Exception as e:
                print(f"Warning: Could not save feature importance: {e}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        if os.path.exists(filepath):
            os.remove(filepath)
        
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
        import traceback
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
        
        return jsonify({
            'success': False,
            'message': 'An error occurred during prediction'
        }), 500

@app.route('/api/prediction-results/<int:job_id>', methods=['GET'])
def get_prediction_results(job_id):
    """Get prediction results for a specific job"""
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        user_id = session.get('user_id')
        cursor.execute(
            "SELECT * FROM prediction_jobs WHERE jobID=%s AND userID=%s",
            (job_id, user_id)
        )
        job = cursor.fetchone()
        
        if not job:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Job not found'}), 404
        
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
        
        cursor.execute(
            """SELECT feature_name, importance_score 
               FROM feature_importance 
               WHERE jobID=%s 
               ORDER BY rank ASC 
               LIMIT 10""",
            (job_id,)
        )
        features = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'job': {
                'id': job['jobID'],
                'filename': job['filename'],
                'status': job['status'],
                'created_at': job['createdAt'].isoformat() if job['createdAt'] else None
            },
            'statistics': {
                'total': job['total_records'],
                'high_risk': job['high_risk_count'],
                'medium_risk': job['medium_risk_count'],
                'low_risk': job['low_risk_count'],
                'avg_score': float(job['average_churn_score'])
            },
            'results': predictions,
            'top_features': [{'name': f['feature_name'], 'importance': float(f['importance_score'])} for f in features],
            'timestamp': job['completedAt'].isoformat() if job['completedAt'] else job['createdAt'].isoformat()
        }), 200
        
    except Exception as e:
        print(f"Error fetching results: {e}")
        return jsonify({'success': False, 'message': 'Error fetching results'}), 500

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
        
        return jsonify({
            'success': True,
            'history': history
        }), 200
        
    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({'success': False, 'message': 'Error fetching history'}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout user"""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'}), 200

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

if __name__ == '__main__':
    print("="*50)
    print("Starting Flask Application")
    print("="*50)
    if MODEL is not None:
        print("✅ Churn prediction model loaded")
    else:
        print("⚠️  Churn prediction model not loaded")
    print("="*50)
    app.run(debug=True, port=5000)