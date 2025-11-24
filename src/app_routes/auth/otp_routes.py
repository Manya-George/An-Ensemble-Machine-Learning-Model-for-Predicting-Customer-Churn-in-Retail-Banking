from flask import Blueprint, request, jsonify, session
import bcrypt
from datetime import datetime, timedelta
import traceback
from core.utils import generate_otp, send_otp_email, log_system_event
from core.helpers import get_db_connection

otp_bp = Blueprint("otp_bp", __name__)

@otp_bp.route('/api/verify-otp', methods=['POST'])
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

@otp_bp.route('/api/resend-otp', methods=['POST'])
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

@otp_bp.route('/api/mark-otp-expired', methods=['POST'])
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