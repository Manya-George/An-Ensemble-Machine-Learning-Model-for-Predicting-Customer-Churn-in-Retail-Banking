from flask import Blueprint, request, jsonify, session
from core.helpers import get_db_connection
from app_routes.auth.auth_routes import userauth_bp
import json
from datetime import datetime, timedelta
import traceback

requests_bp = Blueprint("requests_bp", __name__)

@requests_bp.route('/api/submit-prediction-request', methods=['POST'])
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


@requests_bp.route('/api/pending-requests', methods=['GET'])
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


@requests_bp.route('/api/resolve-request', methods=['POST'])
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


@requests_bp.route('/api/my-requests', methods=['GET'])
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

# ----------------------------
# Prediction Requests API (Retail <> IT workflow)
# ----------------------------

@requests_bp.route('/api/prediction-requests', methods=['POST'])
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


@requests_bp.route('/api/prediction-requests', methods=['GET'])
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


@requests_bp.route('/api/prediction-requests/<int:request_id>/resolve', methods=['POST'])
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
