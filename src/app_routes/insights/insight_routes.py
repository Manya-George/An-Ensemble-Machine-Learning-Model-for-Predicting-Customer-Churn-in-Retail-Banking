from flask import Blueprint, request, jsonify, session
from core.utils import login_required, log_system_event
from core.helpers import get_db_connection
from core.config import UPLOAD_FOLDER
from core.ml_engine import (MODEL, PREPROCESSOR, FEATURE_ENGINEER, EXPLAINER, 
                            _get_top_risk_factors_from_shap, compute_shap_for_single_row)
from datetime import datetime, timedelta
import traceback
from collections import Counter
import pandas as pd
import numpy as np
import json
import math
import os

insights_bp = Blueprint("insights_bp", __name__)

@insights_bp.route('/api/insights-requests', methods=['POST'])
def create_insights_request():
    """Retail Banking user creates an insights request for a prediction job."""
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401

        data = request.json or {}
        description = data.get('description', '').strip()
        job_id = data.get('jobID')

        if not description:
            return jsonify({'success': False, 'message': 'Description is required'}), 400
        if not job_id:
            return jsonify({'success': False, 'message': 'jobID is required'}), 400

        requester_id = session.get('user_id')
        requester_username = session.get('username')

        conn = get_db_connection()
        cursor = conn.cursor()

        # Validate jobID exists
        cursor.execute("SELECT jobID FROM prediction_job WHERE jobID = %s", (job_id,))
        if not cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Invalid jobID'}), 400

        cursor.execute("""
            INSERT INTO insights_requests
                (requester_userID, requester_username, jobID, description)
            VALUES (%s, %s, %s, %s)
        """, (requester_id, requester_username, job_id, description))

        conn.commit()
        insight_id = cursor.lastrowid

        cursor.close()
        conn.close()

        return jsonify({
            'success': True,
            'message': 'Insights request created',
            'insightID': insight_id
        }), 201

    except Exception as e:
        print("Error creating insights request:", e)
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error'}), 500

@insights_bp.route('/api/my-insights-requests', methods=['GET'])
def my_insights_requests():
    """Retail Banking user views only their insight requests."""
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401

        user_id = session.get('user_id')

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT insightID, description, jobID, status, requestedAt, resolvedAt, resolvedByUsername
            FROM insights_requests
            WHERE requester_userID = %s
            ORDER BY requestedAt DESC
        """, (user_id,))

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        for r in rows:
            if isinstance(r.get('requestedAt'), datetime):
                r['requestedAt'] = r['requestedAt'].isoformat()
            if isinstance(r.get('resolvedAt'), datetime):
                r['resolvedAt'] = r['resolvedAt'].isoformat()

        return jsonify({'success': True, 'requests': rows}), 200

    except Exception as e:
        print("my_insights_requests error:", e)
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error'}), 500

@insights_bp.route('/api/insights-requests', methods=['GET'])
def list_insights_requests():
    """
    Business Analysts get all requests.
    Retail users only see their own.
    Supports ?status=pending|resolved or ?mine=1
    """
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401

        role = session.get('role')
        user_id = session.get('user_id')

        status = request.args.get('status', 'all')
        mine = request.args.get('mine')

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        query = "SELECT * FROM insights_requests"
        conditions = []
        params = []

        if status in ("pending", "resolved"):
            conditions.append("status = %s")
            params.append(status)

        if mine:
            conditions.append("requester_userID = %s")
            params.append(user_id)
        else:
            if role != 'Business analyst':
                conditions.append("requester_userID = %s")
                params.append(user_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY requestedAt DESC"

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        for r in rows:
            r['_canResolve'] = (role == 'Business analyst')
            if isinstance(r.get('requestedAt'), datetime):
                r['requestedAt'] = r['requestedAt'].isoformat()
            if isinstance(r.get('resolvedAt'), datetime):
                r['resolvedAt'] = r['resolvedAt'].isoformat()

        return jsonify({'success': True, 'requests': rows}), 200

    except Exception as e:
        print("Error listing insights requests:", e)
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error'}), 500

@insights_bp.route('/api/insights-requests/<int:insight_id>/resolve', methods=['POST'])
def resolve_insights_request(insight_id):
    """Business Analyst resolves an insights request."""
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401

        if session.get('role') != 'Business analyst':
            return jsonify({'success': False, 'message': 'Access denied'}), 403

        data = request.json or {}
        job_id = data.get('jobID')

        if not job_id:
            return jsonify({'success': False, 'message': 'jobID is required'}), 400

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Check that request exists
        cursor.execute("SELECT * FROM insights_requests WHERE insightID = %s", (insight_id,))
        req = cursor.fetchone()
        if not req:
            return jsonify({'success': False, 'message': 'Request not found'}), 404

        # Validate job exists
        cursor.execute("SELECT jobID FROM prediction_job WHERE jobID = %s", (job_id,))
        if not cursor.fetchone():
            return jsonify({'success': False, 'message': 'Invalid jobID'}), 400

        resolver_id = session.get('user_id')
        resolver_username = session.get('username')

        cursor.execute("""
            UPDATE insights_requests
            SET status='resolved',
                resolvedAt = NOW(),
                resolvedBy = %s,
                resolvedByUsername = %s
            WHERE insightID = %s
        """, (resolver_id, resolver_username, insight_id))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({'success': True, 'message': 'Request resolved'}), 200

    except Exception as e:
        print("Error resolving insights request:", e)
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error'}), 500

@insights_bp.route('/api/ba/prediction-insights/<int:job_id>', methods=['GET'])
def ba_prediction_insights(job_id):
    """
    FIXED: BA endpoint using CSV-scoping logic like get_prediction_results.
    Loads the specific CSV file for the job and processes only those customers.
    Supports filtering by risk factors, risk level, search, and pagination.
    """
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401

        role = session.get('role')
        if role not in ('Business analyst',):
            return jsonify({'success': False, 'message': 'Access denied'}), 403

        # Query params for filtering
        factor_filter = request.args.get('factor')
        risk_filter = request.args.get('risk', 'all').lower()
        search = (request.args.get('search') or '').strip()

        try:
            page = max(1, int(request.args.get('page', 1)))
        except:
            page = 1
        try:
            per_page = max(10, min(500, int(request.args.get('per_page', 50))))
        except:
            per_page = 50

        sort = request.args.get('sort', 'prob_desc')

        # ========================================
        # STEP 1: Fetch job metadata and CSV path
        # ========================================
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT jobID, filename, saved_filename, status, processed_records,
                   high_risk_count, medium_risk_count, low_risk_count,
                   average_churn_score
            FROM prediction_job
            WHERE jobID=%s
        """, (job_id,))
        job = cursor.fetchone()

        if not job:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Job not found'}), 404

        saved_filename = job.get('saved_filename')
        if not saved_filename:
            cursor.close()
            conn.close()
            return jsonify({
                'success': False,
                'message': 'No saved file found for this job'
            }), 404

        # Clean the path
        clean_filename = saved_filename.replace("uploads/", "").replace("uploads\\", "")
        csv_path = os.path.join(UPLOAD_FOLDER, clean_filename) if not saved_filename.startswith(UPLOAD_FOLDER) else saved_filename

        if not os.path.exists(csv_path):
            cursor.close()
            conn.close()
            return jsonify({
                'success': False,
                'message': f'Saved CSV file not found: {clean_filename}'
            }), 404

        print(f"📂 BA loading CSV for job {job_id}: {csv_path}")

        # Load the CSV file for THIS job
        df_original = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df_original)} rows from CSV")

        # ========================================
        # STEP 2: Fetch DB predictions for this job
        # ========================================
        cursor.execute("""
            SELECT predictionID, jobID, customerID, surname,
                   credit_score, geography, gender, age, tenure, balance,
                   num_of_products, has_credit_card, is_active_member, estimated_salary,
                   churn_probability, churn_prediction, risk_level, top_risk_factors, createdAt
            FROM churn_predictions
            WHERE jobID=%s
            ORDER BY predictionID ASC
        """, (job_id,))
        db_predictions = cursor.fetchall()

        cursor.close()
        conn.close()

        print(f"🔍 Found {len(db_predictions)} predictions in DB for job {job_id}")

        if not db_predictions:
            return jsonify({'success': False, 'message': 'No predictions found'}), 404

        # ========================================
        # STEP 3: Build lookup map from DB predictions
        # ========================================
        db_map = {}
        for pred in db_predictions:
            cust_id = str(pred.get('customerID'))
            db_map[cust_id] = pred

        # ========================================
        # STEP 4: Process CSV rows with SHAP computation
        # ========================================
        processed_rows = []
        overall_factor_counter = Counter()
        rows_with_shap = 0
        rows_needing_shap = 0

        # Iterate through CSV rows (CSV is source of truth!)
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

            # Parse stored top_risk_factors
            parsed_factors = []
            if db_pred.get("top_risk_factors"):
                try:
                    parsed_factors = json.loads(db_pred["top_risk_factors"])
                    if parsed_factors and len(parsed_factors) > 0:
                        rows_with_shap += 1
                except Exception as e:
                    print(f"⚠️  Failed to parse stored factors for {customer_id}: {e}")
                    parsed_factors = []

            # Compute SHAP if missing
            if not parsed_factors or len(parsed_factors) == 0:
                rows_needing_shap += 1
                try:
                    # Use CSV row data for SHAP computation
                    raw_row_dict = csv_row.to_dict()
                    parsed_factors = compute_shap_for_single_row(raw_row_dict)

                    if parsed_factors and len(parsed_factors) > 0:
                        print(f"✅ Computed SHAP for customer {customer_id} (row {idx})")
                    else:
                        print(f"⚠️  SHAP returned empty for customer {customer_id}")

                except Exception as e:
                    print(f"❌ SHAP computation failed for customer {customer_id}: {e}")
                    parsed_factors = []

            # Extract factor names
            factor_names = [
                item["factor"] for item in parsed_factors
                if isinstance(item, dict) and "factor" in item
            ]

            # Count factors
            for fn in factor_names:
                overall_factor_counter[fn] += 1

            # Build enriched row
            enriched_row = {
                "predictionID": db_pred["predictionID"],
                "customerID": customer_id,
                "surname": db_pred.get("surname") or csv_row.get("Surname"),
                "credit_score": db_pred.get("credit_score"),
                "geography": db_pred.get("geography"),
                "gender": db_pred.get("gender"),
                "age": db_pred.get("age"),
                "tenure": db_pred.get("tenure"),
                "balance": db_pred.get("balance"),
                "num_of_products": db_pred.get("num_of_products"),
                "has_credit_card": db_pred.get("has_credit_card"),
                "is_active_member": db_pred.get("is_active_member"),
                "estimated_salary": db_pred.get("estimated_salary"),
                "churn_probability": float(db_pred.get("churn_probability") or 0.0),
                "churn_prediction": bool(db_pred.get("churn_prediction")),
                "risk_level": db_pred.get("risk_level"),
                "createdAt": db_pred.get("createdAt"),
                "_parsed_top_risk_factors": parsed_factors,
                "_top_factor_names": factor_names
            }
            processed_rows.append(enriched_row)

        print(f"📊 Processed {len(processed_rows)} customers from CSV")
        print(f"   SHAP already in DB: {rows_with_shap}")
        print(f"   SHAP computed on-demand: {rows_needing_shap}")

        # ========================================
        # STEP 5: Apply filters
        # ========================================
        filtered = processed_rows

        # Filter by risk factor
        if factor_filter:
            f_lower = factor_filter.lower()
            filtered = [
                r for r in filtered
                if any(f_lower == fn.lower() for fn in r["_top_factor_names"])
            ]
            print(f"🔍 Filtered by factor '{factor_filter}': {len(filtered)} customers")

        # Filter by risk level
        if risk_filter in ("low", "medium", "high"):
            filtered = [
                r for r in filtered
                if (r.get("risk_level") or "").lower() == risk_filter
            ]
            print(f"🔍 Filtered by risk '{risk_filter}': {len(filtered)} customers")

        # Filter by search (customerID or surname)
        if search:
            s = search.lower()
            filtered = [
                r for r in filtered
                if s in str(r.get("customerID", "")).lower()
                or (r.get("surname") and s in r["surname"].lower())
            ]
            print(f"🔍 Filtered by search '{search}': {len(filtered)} customers")

        # Factor counts for filtered set
        filtered_factor_counter = Counter()
        for r in filtered:
            for f in r["_top_factor_names"]:
                filtered_factor_counter[f] += 1

        # ========================================
        # STEP 6: Sorting
        # ========================================
        if sort == "prob_desc":
            filtered.sort(key=lambda x: float(x["churn_probability"]), reverse=True)
        elif sort == "prob_asc":
            filtered.sort(key=lambda x: float(x["churn_probability"]))
        elif sort == "id":
            filtered.sort(key=lambda x: int(x["customerID"]) if str(x["customerID"]).isdigit() else str(x["customerID"]))
        elif sort == "risk":
            order = {"high": 0, "medium": 1, "low": 2}
            filtered.sort(key=lambda x: order.get((x["risk_level"] or "").lower(), 3))

        # ========================================
        # STEP 7: Pagination
        # ========================================
        total = len(filtered)
        total_pages = max(1, math.ceil(total / per_page))

        page = min(page, total_pages)
        start = (page - 1) * per_page
        end = start + per_page
        page_rows = filtered[start:end]

        # ========================================
        # STEP 8: Build response
        # ========================================
        result_rows = []
        for r in page_rows:
            result_rows.append({
                "predictionID": r["predictionID"],
                "customer_id": r["customerID"],
                "surname": r["surname"],
                "credit_score": r["credit_score"],
                "geography": r["geography"],
                "gender": r["gender"],
                "age": r["age"],
                "tenure": r["tenure"],
                "balance": float(r["balance"]) if r["balance"] is not None else None,
                "num_of_products": r["num_of_products"],
                "has_credit_card": bool(r["has_credit_card"]),
                "is_active_member": bool(r["is_active_member"]),
                "estimated_salary": float(r["estimated_salary"]) if r["estimated_salary"] is not None else None,
                "churn_probability": float(r["churn_probability"]),
                "churn_prediction": bool(r["churn_prediction"]),
                "risk_level": r["risk_level"],
                "top_risk_factors": r["_parsed_top_risk_factors"],
                "top_factor_names": r["_top_factor_names"],
                "createdAt": r["createdAt"].isoformat() if isinstance(r["createdAt"], datetime) else r["createdAt"],
            })

        return jsonify({
            "success": True,
            "job_id": job_id,
            "filename": job.get('filename'),
            "csv_row_count": len(df_original),
            "total_predictions": len(processed_rows),
            "total_filtered": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "rows": result_rows,
            "available_factors": list(overall_factor_counter.keys()),
            "overall_factor_counts": dict(overall_factor_counter),
            "filtered_factor_counts": dict(filtered_factor_counter),
            "applied_filters": {
                "factor": factor_filter,
                "risk": risk_filter,
                "search": search,
                "sort": sort
            },
            "statistics": {
                "high_risk": sum(1 for r in processed_rows if r["risk_level"] == "high"),
                "medium_risk": sum(1 for r in processed_rows if r["risk_level"] == "medium"),
                "low_risk": sum(1 for r in processed_rows if r["risk_level"] == "low"),
                "avg_score": sum(r["churn_probability"] for r in processed_rows) / len(processed_rows) if processed_rows else 0.0
            }
        }), 200

    except Exception as e:
        print(f"❌ BA insights ERROR: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error'}), 500