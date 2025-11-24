from flask import Blueprint, request, jsonify, session
from core.helpers import get_db_connection
from core.config import UPLOAD_FOLDER
from core.ml_engine import (MODEL, PREPROCESSOR, EXPLAINER, _get_top_risk_factors_from_shap, 
                            FEATURE_ENGINEER, compute_shap_for_single_row, _build_original_row_from_db)
from datetime import datetime, timedelta
import traceback
from app_routes.auth.auth_routes import userauth_bp
import numpy as np
import pandas as pd
import json
import os

results_bp = Blueprint("results_bp", __name__)

@results_bp.route('/api/prediction-results/<int:job_id>', methods=['GET'])
def get_prediction_results(job_id):
    """
    FIXED: Returns prediction results for a SPECIFIC job using its saved CSV file.
    - Fetches job metadata and saved_filename from DB
    - Loads the EXACT CSV file for that job from uploads folder
    - Computes/retrieves SHAP for ONLY those customers in that file
    """
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Fetch job metadata INCLUDING saved_filename
        cursor.execute("""
            SELECT jobID, filename, saved_filename, status, completedAt, processed_records,
                   high_risk_count, medium_risk_count, low_risk_count,
                   average_churn_score
            FROM prediction_job
            WHERE jobID=%s
        """, (job_id,))
        job = cursor.fetchone()
        
        if not job:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Job ID not found'}), 404

        saved_filename = job.get('saved_filename')
        if not saved_filename:
            cursor.close()
            conn.close()
            return jsonify({
                'success': False, 
                'message': 'No saved file found for this job'
            }), 404

        # Clean the path (remove any 'uploads/' prefix if already in path)
        clean_filename = saved_filename.replace("uploads/", "").replace("uploads\\", "")
        csv_path = os.path.join(UPLOAD_FOLDER, clean_filename) if not saved_filename.startswith(UPLOAD_FOLDER) else saved_filename
        
        if not os.path.exists(csv_path):
            cursor.close()
            conn.close()
            return jsonify({
                'success': False,
                'message': f'Saved CSV file not found: {clean_filename}'
            }), 404

        print(f"📂 Loading CSV for job {job_id}: {csv_path}")
        
        # Load the EXACT CSV file for this job
        df_original = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df_original)} rows from CSV")

        # Fetch predictions for THIS job from database
        cursor.execute("""
            SELECT predictionID, customerID, surname,
                   credit_score, geography, gender, age, tenure, balance,
                   num_of_products, has_credit_card, is_active_member,
                   estimated_salary,
                   churn_probability, churn_prediction,
                   risk_level, top_risk_factors
            FROM churn_predictions
            WHERE jobID=%s
            ORDER BY predictionID ASC
        """, (job_id,))
        db_predictions = cursor.fetchall()

        cursor.close()
        conn.close()

        print(f"🔍 Found {len(db_predictions)} predictions in DB for job {job_id}")

        # Build results using CSV data + DB predictions
        results = []
        rows_needing_shap = 0
        rows_with_shap = 0

        # Create a mapping of customerID -> DB prediction for fast lookup
        db_map = {}
        for pred in db_predictions:
            cust_id = str(pred.get('customerID'))
            db_map[cust_id] = pred

        # Process each row from the CSV file (THIS IS THE KEY FIX!)
        for idx, csv_row in df_original.iterrows():
            # Get customer ID from CSV
            custid_raw = (csv_row.get('CustomerId') or csv_row.get('customerID') or 
                         csv_row.get('customerId') or csv_row.get('CustomerID'))
            customer_id = str(custid_raw) if custid_raw is not None else f"CUST_{idx}"

            # Find corresponding DB prediction
            db_pred = db_map.get(customer_id)
            
            if not db_pred:
                print(f"⚠️  No DB prediction found for customer {customer_id} (row {idx})")
                continue

            # Get stored top_risk_factors
            stored_factors = []
            if db_pred.get("top_risk_factors"):
                try:
                    stored_factors = json.loads(db_pred["top_risk_factors"])
                    if stored_factors and len(stored_factors) > 0:
                        rows_with_shap += 1
                except Exception as e:
                    print(f"⚠️  Failed to parse stored factors for {customer_id}: {e}")
                    stored_factors = []

            # Compute SHAP only if missing
            computed_factors = stored_factors
            if not stored_factors or len(stored_factors) == 0:
                rows_needing_shap += 1
                try:
                    # Use the CSV row data for SHAP computation
                    raw_row_dict = csv_row.to_dict()
                    computed_factors = compute_shap_for_single_row(raw_row_dict)
                    
                    if computed_factors and len(computed_factors) > 0:
                        print(f"✅ Computed SHAP for customer {customer_id} (row {idx})")
                    else:
                        print(f"⚠️  SHAP returned empty for customer {customer_id}")
                        
                except Exception as e:
                    print(f"❌ SHAP computation failed for customer {customer_id}: {e}")
                    computed_factors = []

            # Build result record
            result = {
                "customer_id": customer_id,
                "surname": db_pred.get("surname") or csv_row.get("Surname"),
                "churn_probability": float(db_pred.get("churn_probability") or 0.0),
                "churn_prediction": bool(db_pred.get("churn_prediction")),
                "risk_level": db_pred.get("risk_level"),
                "top_risk_factors": computed_factors if computed_factors else []
            }
            results.append(result)

        # Sort by probability descending
        results.sort(key=lambda x: x["churn_probability"], reverse=True)

        print(f"📊 Results summary for job {job_id}:")
        print(f"   CSV rows: {len(df_original)}")
        print(f"   DB predictions: {len(db_predictions)}")
        print(f"   Final results: {len(results)}")
        print(f"   SHAP already in DB: {rows_with_shap}")
        print(f"   SHAP computed on-demand: {rows_needing_shap}")

        # Response
        return jsonify({
            "success": True,
            "job_id": job_id,
            "filename": job['filename'],
            "saved_filename": saved_filename,
            "timestamp": job["completedAt"].isoformat() if job["completedAt"] else None,
            "statistics": {
                "total": len(results),  # Use actual result count
                "high_risk": int(job.get("high_risk_count") or sum(1 for r in results if r["risk_level"] == "high")),
                "medium_risk": int(job.get("medium_risk_count") or sum(1 for r in results if r["risk_level"] == "medium")),
                "low_risk": int(job.get("low_risk_count") or sum(1 for r in results if r["risk_level"] == "low")),
                "avg_score": float(job.get("average_churn_score") or (
                    sum(r["churn_probability"] for r in results) / len(results) if results else 0.0
                ))
            },
            "results": results
        }), 200

    except Exception as e:
        print(f"❌ ERROR in get_prediction_results: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


@results_bp.route('/api/prediction-history', methods=['GET'])
def get_prediction_history():
    """Get user's prediction history."""
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        user_id = session.get('user_id')
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(
            """SELECT 
                jobID, filename, status, total_records, processed_records,
                high_risk_count, medium_risk_count, low_risk_count, 
                average_churn_score, createdAt, completedAt
                FROM prediction_job 
                WHERE userID=%s 
                ORDER BY createdAt DESC
                LIMIT 50""",
            (user_id,)
        )
        history = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        print(f"📜 Fetched {len(history)} history records for user {user_id}")
        
        return jsonify({
            'success': True,
            'history': history
        }), 200
        
    except Exception as e:
        print(f"❌ Error fetching history: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Error fetching history'}), 500


@results_bp.route('/api/debug/job-details/<int:job_id>', methods=['GET'])
def debug_job_details(job_id):
    """
    DEBUG ENDPOINT: Get detailed information about a specific job and its predictions.
    Helps identify if results are being mixed between jobs.
    """
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get job info
        cursor.execute("""
            SELECT jobID, userID, filename, saved_filename, status, total_records, processed_records,
                   high_risk_count, medium_risk_count, low_risk_count, 
                   average_churn_score, createdAt, completedAt
            FROM prediction_job
            WHERE jobID=%s
        """, (job_id,))
        job_info = cursor.fetchone()
        
        if not job_info:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Job not found'}), 404
        
        # Count predictions for THIS job
        cursor.execute("""
            SELECT COUNT(*) as count FROM churn_predictions WHERE jobID=%s
        """, (job_id,))
        prediction_count = cursor.fetchone()['count']
        
        # Get sample predictions (first 5)
        cursor.execute("""
            SELECT predictionID, customerID, jobID, churn_probability, risk_level
            FROM churn_predictions
            WHERE jobID=%s
            ORDER BY predictionID ASC
            LIMIT 5
        """, (job_id,))
        sample_predictions = cursor.fetchall()
        
        # Check if CSV file exists
        saved_filename = job_info.get('saved_filename')
        csv_exists = False
        csv_row_count = 0
        
        if saved_filename:
            clean_filename = saved_filename.replace("uploads/", "").replace("uploads\\", "")
            csv_path = os.path.join(UPLOAD_FOLDER, clean_filename) if not saved_filename.startswith(UPLOAD_FOLDER) else saved_filename
            csv_exists = os.path.exists(csv_path)
            
            if csv_exists:
                try:
                    df = pd.read_csv(csv_path)
                    csv_row_count = len(df)
                except Exception as e:
                    print(f"Error reading CSV: {e}")
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'job_info': job_info,
            'prediction_count': prediction_count,
            'sample_predictions': sample_predictions,
            'csv_file': {
                'exists': csv_exists,
                'path': saved_filename,
                'row_count': csv_row_count
            },
            'diagnosis': {
                'records_match': job_info['processed_records'] == prediction_count,
                'csv_matches_db': csv_row_count == prediction_count if csv_exists else None,
                'expected': job_info['processed_records'],
                'actual_db': prediction_count,
                'actual_csv': csv_row_count if csv_exists else 0
            }
        }), 200
        
    except Exception as e:
        print(f"❌ Debug error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500