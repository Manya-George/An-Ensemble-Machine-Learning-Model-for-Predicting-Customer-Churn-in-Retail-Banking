from flask import Blueprint, request, jsonify, session
from werkzeug.utils import secure_filename
from core.config import UPLOAD_FOLDER
from core.helpers import get_db_connection, allowed_file, calculate_risk_level
from core.ml_engine import (MODEL, PREPROCESSOR, EXPLAINER, compute_shap_for_single_row,
                            _get_top_risk_factors_from_shap, FEATURE_ENGINEER, THRESHOLD)
from datetime import datetime, timedelta
import time
import traceback
import os
import pandas as pd
import numpy as np
import json

upload_bp = Blueprint("upload_bp", __name__)

@upload_bp.route('/api/predict-churn', methods=['POST'])
def predict_churn():
    """
    FIXED: Prediction route that computes SHAP for EVERY customer and stores top_risk_factors.
    """
    conn = None
    cursor = None
    filepath = None
    job_id = None

    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401

        if MODEL is None or PREPROCESSOR is None or FEATURE_ENGINEER is None or EXPLAINER is None:
            return jsonify({'success': False, 'message': 'Prediction model not fully loaded.'}), 503

        # File checks
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': 'Invalid file type. Please upload CSV'}), 400

        original_filename = secure_filename(file.filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        saved_filename = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{original_filename}")
        file.save(saved_filename)

        df_raw = pd.read_csv(saved_filename)
        if df_raw.shape[0] == 0:
            return jsonify({'success': False, 'message': 'Uploaded file is empty'}), 400

        print(f"📁 Processing {len(df_raw)} rows from {original_filename}")

        # DB setup and insert job record
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = session.get('user_id')
        cursor.execute(
            """INSERT INTO prediction_job (userID, filename, saved_filename, status, total_records)
               VALUES (%s, %s, %s, 'processing', %s)""",
            (user_id, original_filename, saved_filename, int(len(df_raw)))
        )
        conn.commit()
        job_id = int(cursor.lastrowid)

        print(f"✅ Created job ID: {job_id}")

        # Feature engineering & preprocessing (BATCH)
        df_engineered = FEATURE_ENGINEER.transform(df_raw.copy())
        X_processed = PREPROCESSOR.transform(df_engineered)

        # Predictions (BATCH)
        predictions = MODEL.predict(X_processed)
        probabilities = MODEL.predict_proba(X_processed)[:, 1]

        print(f"🔮 Generated predictions for {len(probabilities)} rows")

        # SHAP computation (BATCH) - we'll compute all at once then map to rows
        print(f"🧠 Computing SHAP values for all {len(X_processed)} rows...")
        shap_values = None
        try:
            shap_raw = EXPLAINER.shap_values(X_processed)
            # Handle both binary classifier output formats
            if isinstance(shap_raw, list) and len(shap_raw) > 1:
                shap_values = np.array(shap_raw[1])  # Class 1 (churn)
            else:
                shap_values = np.array(shap_raw)
            print(f"✅ SHAP computed successfully: shape {shap_values.shape}")
        except Exception as e:
            print(f"❌ SHAP GENERATION FAILED: {e}")
            traceback.print_exc()
            shap_values = None

        # Get processed feature names for SHAP mapping
        feature_names_processed = list(PREPROCESSOR.get_feature_names_out())

        # Stats calculation
        probs = np.array(probabilities)
        
        high_risk = sum(1 for p in probabilities if p >= 0.7)
        medium_risk = sum(1 for p in probabilities if 0.4 <= p < 0.7)
        low_risk = sum(1 for p in probabilities if p < 0.4)
        avg_score = np.mean(probabilities)
        total_records = int(len(df_raw))

        print(f"📊 Risk distribution: High={high_risk}, Medium={medium_risk}, Low={low_risk}")

        # Update job completed metadata
        cursor.execute(
            """UPDATE prediction_job SET status='completed', processed_records=%s,
               high_risk_count=%s, medium_risk_count=%s, low_risk_count=%s,
               average_churn_score=%s, completedAt=NOW()
               WHERE jobID=%s""",
            (total_records, high_risk, medium_risk, low_risk, avg_score, job_id)
        )
        conn.commit()

        # Helper function for safe type conversion
        def safe_py(val, typ=None, default=None):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            if isinstance(val, (np.generic,)):
                return val.item()
            if typ == int:
                try: return int(val)
                except: return default if default is not None else 0
            if typ == float:
                try: return float(val)
                except: return default if default is not None else 0.0
            if typ == bool:
                return bool(val)
            return val

        # Build per-row insert tuples with SHAP for EVERY row
        predictions_to_insert = []
        seen_customers = set()
        BATCH_SIZE = 500  # Reduced for safety with JSON columns
        
        print(f"🔄 Processing individual rows and computing SHAP...")

        for idx, row in df_raw.iterrows():
            if idx % 100 == 0:
                print(f"   Processing row {idx}/{len(df_raw)}...")

            # Customer ID for dedupe (case-insensitive)
            custid_raw = row.get('CustomerId') or row.get('customerID') or row.get('customerId') or row.get('CustomerID')
            customer_id = str(custid_raw) if custid_raw is not None else f"CUST_{idx}"
            
            # Deduplicate - keep first occurrence
            if customer_id in seen_customers:
                continue
            seen_customers.add(customer_id)

            probability = float(probs[idx])
            pred_bool = bool(predictions[idx])
            risk_level = calculate_risk_level(probability)

            # CRITICAL FIX: Compute SHAP for THIS specific row
            top_risk_factors_list = []
            
            if shap_values is not None and idx < len(shap_values):
                try:
                    # Get SHAP values for this specific row
                    shap_row = shap_values[idx]
                    
                    # Build raw data dict for this row
                    raw_dict = {k: safe_py(v) for k, v in row.to_dict().items()}
                    
                    # Get top risk factors using the row's SHAP values
                    top_risk_factors_list = _get_top_risk_factors_from_shap(
                        shap_row,
                        feature_names_processed,
                        raw_dict,
                        top_n=3
                    )
                    
                    if not top_risk_factors_list:
                        print(f"⚠️  Row {idx}: SHAP returned empty factors, trying single-row compute...")
                        # Fallback to single-row computation
                        top_risk_factors_list = compute_shap_for_single_row(raw_dict)
                        
                except Exception as e:
                    print(f"❌ Row {idx}: SHAP factor extraction failed: {e}")
                    # Try alternative single-row computation
                    try:
                        raw_dict = {k: safe_py(v) for k, v in row.to_dict().items()}
                        top_risk_factors_list = compute_shap_for_single_row(raw_dict)
                    except Exception as e2:
                        print(f"❌ Row {idx}: Fallback SHAP also failed: {e2}")
                        top_risk_factors_list = []
            else:
                # SHAP batch failed, compute individually
                try:
                    raw_dict = {k: safe_py(v) for k, v in row.to_dict().items()}
                    top_risk_factors_list = compute_shap_for_single_row(raw_dict)
                except Exception as e:
                    print(f"❌ Row {idx}: Individual SHAP computation failed: {e}")
                    top_risk_factors_list = []

            # Convert to JSON
            top_risk_factors_json = json.dumps(top_risk_factors_list, ensure_ascii=False)

            # Build insert tuple
            tup = (
                int(job_id),
                customer_id,
                safe_py(row.get('Surname'), str, None),
                safe_py(row.get('CreditScore'), int, None),
                safe_py(row.get('Geography'), str, ''),
                safe_py(row.get('Gender'), str, ''),
                safe_py(row.get('Age'), int, None),
                safe_py(row.get('Tenure'), int, None),
                safe_py(row.get('Balance'), float, 0.0),
                safe_py(row.get('NumOfProducts'), int, 0),
                1 if safe_py(row.get('HasCrCard'), int, 0) else 0,
                1 if safe_py(row.get('IsActiveMember'), int, 0) else 0,
                safe_py(row.get('EstimatedSalary'), float, 0.0),
                float(probability),
                1 if pred_bool else 0,
                risk_level,
                top_risk_factors_json
            )
            predictions_to_insert.append(tup)

            # Batch insert when full
            if len(predictions_to_insert) >= BATCH_SIZE:
                cursor.executemany(
                    """INSERT INTO churn_predictions
                       (jobID, customerID, surname, credit_score, geography, gender,
                        age, tenure, balance, num_of_products, has_credit_card,
                        is_active_member, estimated_salary, churn_probability,
                        churn_prediction, risk_level, top_risk_factors)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    predictions_to_insert
                )
                conn.commit()
                print(f"   ✅ Inserted batch of {len(predictions_to_insert)} records")
                predictions_to_insert = []

        # Final batch
        if predictions_to_insert:
            cursor.executemany(
                """INSERT INTO churn_predictions
                   (jobID, customerID, surname, credit_score, geography, gender,
                    age, tenure, balance, num_of_products, has_credit_card,
                    is_active_member, estimated_salary, churn_probability,
                    churn_prediction, risk_level, top_risk_factors)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                predictions_to_insert
            )
            conn.commit()
            print(f"   ✅ Inserted final batch of {len(predictions_to_insert)} records")

        cursor.close()
        conn.close()

        print(f"✅ Job {job_id} completed successfully!")

        return jsonify({
            'success': True,
            'message': 'Prediction completed successfully',
            'job_id': job_id,
            'statistics': {
                'total': total_records,
                'high_risk': high_risk,
                'medium_risk': medium_risk,
                'low_risk': low_risk,
                'avg_score': avg_score
            }
        }), 200

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        try:
            if conn and job_id is not None:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE prediction_job SET status='failed', error_message=%s WHERE jobID=%s",
                    (str(e), int(job_id))
                )
                conn.commit()
                cursor.close()
                conn.close()
        except Exception:
            pass

        return jsonify({
            'success': False,
            'message': f'An error occurred: {str(e)}'
        }), 500


@upload_bp.route('/api/predict', methods=['POST'])
def predict_json():
    """
    JSON-based prediction endpoint (real-time or batch).
    """
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401

        if MODEL is None or PREPROCESSOR is None:
            return jsonify({'success': False, 'message': 'Prediction model not loaded'}), 500

        payload = request.get_json()
        if payload is None:
            return jsonify({'success': False, 'message': 'No JSON payload received'}), 400

        if isinstance(payload, dict):
            df_input = pd.DataFrame([payload])
        elif isinstance(payload, list):
            df_input = pd.DataFrame(payload)
        else:
            return jsonify({'success': False, 'message': 'Invalid payload format'}), 400

        # Feature engineering
        df_engineered = FEATURE_ENGINEER.transform(df_input.copy())
        X_proc = PREPROCESSOR.transform(df_engineered)

        probs = MODEL.predict_proba(X_proc)[:, 1]
        preds = (probs >= THRESHOLD).astype(int)

        # Compute SHAP for each row
        results = []
        for i in range(len(df_input)):
            prob = float(probs[i])
            pred = int(preds[i])
            risk = calculate_risk_level(prob)
            
            # Compute SHAP for this row
            try:
                raw_dict = df_input.iloc[i].to_dict()
                top_feats = compute_shap_for_single_row(raw_dict)
            except Exception as e:
                print(f"SHAP failed for row {i}: {e}")
                top_feats = []

            results.append({
                "customer_id": int(df_input.iloc[i].get("CustomerId", i)),
                "surname": str(df_input.iloc[i].get("Surname", "")),
                "churn_probability": prob,
                "churn_prediction": pred,
                "risk_level": risk,
                "top_risk_factors": top_feats
            })

        high = sum(1 for r in results if r['risk_level'] == 'high')
        med = sum(1 for r in results if r['risk_level'] == 'medium')
        low = sum(1 for r in results if r['risk_level'] == 'low')
        avg = float(np.mean([r['churn_probability'] for r in results])) if results else 0.0

        return jsonify({
            "timestamp": datetime.utcnow().isoformat(),
            "statistics": {
                "total": len(results),
                "high_risk": high,
                "medium_risk": med,
                "low_risk": low,
                "avg_score": avg
            },
            "results": results
        }), 200

    except Exception as e:
        print(f"❌ Error in /api/predict: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Prediction failed'}), 500