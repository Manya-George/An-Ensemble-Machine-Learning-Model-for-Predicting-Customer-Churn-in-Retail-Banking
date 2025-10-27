#churn_prediction_routes.py

from flask import request, jsonify, session
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import mysql.connector
import os
from werkzeug.utils import secure_filename
import uuid

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MODEL_PATH = 'catboost_churn_model.joblib'

# Load the trained model
try:
    model_artifacts = joblib.load(MODEL_PATH)
    MODEL = model_artifacts['model']
    PREPROCESSOR = model_artifacts['preprocessor']
    SELECTOR = model_artifacts['selector']
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='',  # Update with your MySQL password
        database='ChurnDBTest'
    )

def calculate_risk_level(probability):
    """Determine risk level based on churn probability"""
    if probability >= 0.7:
        return 'high'
    elif probability >= 0.4:
        return 'medium'
    else:
        return 'low'

@app.route('/api/predict-churn', methods=['POST'])
def predict_churn():
    """Handle churn prediction for uploaded CSV file"""
    try:
        # Check if user is authenticated
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': 'Invalid file type. Please upload a CSV file'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        job_id = str(uuid.uuid4())
        filepath = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
        file.save(filepath)
        
        # Read CSV
        df = pd.read_csv(filepath)
        
        # Create prediction job record
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
        
        # Prepare data for prediction
        # Assuming your model expects these columns
        required_columns = [
            'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
            'EstimatedSalary'
        ]
        
        # Check if required columns exist
        missing_cols = [col for col in required_columns if col not in df.columns]
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
        
        # Prepare features
        X = df[required_columns].copy()
        
        # Preprocess and predict
        X_processed = PREPROCESSOR.transform(X)
        X_selected = SELECTOR.transform(X_processed)
        
        # Get predictions and probabilities
        predictions = MODEL.predict(X_selected)
        probabilities = MODEL.predict_proba(X_selected)[:, 1]
        
        # Calculate statistics
        high_risk = sum(1 for p in probabilities if p >= 0.7)
        medium_risk = sum(1 for p in probabilities if 0.4 <= p < 0.7)
        low_risk = sum(1 for p in probabilities if p < 0.4)
        avg_score = np.mean(probabilities)
        
        # Update job status
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
        
        # Insert individual predictions
        for idx, row in df.iterrows():
            customer_id = row.get('CustomerId', f'CUST_{idx}')
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
        
        # Store feature importance
        if hasattr(MODEL, 'get_feature_importance'):
            feature_names = SELECTOR.get_feature_names_out() if hasattr(SELECTOR, 'get_feature_names_out') else [f'feature_{i}' for i in range(X_selected.shape[1])]
            importances = MODEL.get_feature_importance()
            
            # Sort and get top features
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
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Clean up uploaded file
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
        
        # Update job status to failed
        try:
            if 'job_db_id' in locals():
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
        
        # Verify job belongs to user
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
        
        # Get predictions
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
        
        # Get feature importance
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