from flask import Blueprint, request, jsonify, session
import bcrypt
from datetime import datetime, timedelta
import traceback
from core.utils import generate_otp, send_otp_email, log_system_event
from core.helpers import get_db_connection

logs_bp = Blueprint("logs_bp", __name__)

@logs_bp.route('/api/recent-logs', methods=['GET'])
def get_recent_logs():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT s.timestamp, u.username, s.action_description, s.status_code
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

@logs_bp.route('/api/system-logs', methods=['GET'])
def get_system_logs():
    """Retrieve recent system logs (IT admin only, with optional filters)"""
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        role = session.get('role')
        if role != 'IT admin':
            return jsonify({'success': False, 'message': 'Access denied'}), 403

        # --- Read optional query parameters for filtering ---
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        user = request.args.get('user', '').strip()
        endpoint = request.args.get('endpoint', '').strip()
        status = request.args.get('status', '').strip()
        search = request.args.get('search', '').strip()

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # --- Build dynamic query safely ---
        query = """
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
            WHERE 1=1
        """
        params = []

        if start_date:
            query += " AND l.timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND l.timestamp <= %s"
            params.append(end_date + " 23:59:59")
        if user:
            query += " AND u.username LIKE %s"
            params.append(f"%{user}%")
        if endpoint:
            query += " AND l.endpoint LIKE %s"
            params.append(f"%{endpoint}%")
        if status:
            query += " AND l.status_code = %s"
            params.append(status)
        if search:
            query += " AND (l.action_description LIKE %s OR l.endpoint LIKE %s)"
            params.extend([f"%{search}%", f"%{search}%"])

        query += " ORDER BY l.timestamp DESC LIMIT 500"

        cursor.execute(query, tuple(params))
        logs = cursor.fetchall()

        cursor.close()
        conn.close()

        return jsonify({'success': True, 'logs': logs}), 200

    except Exception as e:
        print("Error fetching logs:", e)
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Server error'}), 500

