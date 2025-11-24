from flask import Blueprint, request, jsonify, session
import bcrypt
from datetime import datetime, timedelta
import traceback
from core.utils import generate_otp, send_otp_email, log_system_event
from core.helpers import get_db_connection, get_model_artifacts_info
from core.config import APP_START_TIME

dashboard_bp = Blueprint("dashboard_bp", __name__)

@dashboard_bp.route('/api/retail-dashboard-stats', methods=['GET'])
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


@dashboard_bp.route('/api/admin-stats', methods=['GET'])
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

@dashboard_bp.route('/api/system-health', methods=['GET'])
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

@dashboard_bp.route('/api/model-performance', methods=['GET'])
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

@dashboard_bp.route('/api/api-metrics', methods=['GET'])
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

