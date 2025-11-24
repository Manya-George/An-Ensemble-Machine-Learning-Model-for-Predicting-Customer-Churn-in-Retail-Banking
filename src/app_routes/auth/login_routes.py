from flask import Blueprint, request, jsonify, session
import bcrypt
from datetime import datetime, timedelta
import traceback
from core.utils import generate_otp, send_otp_email, log_system_event
from core.helpers import get_db_connection


auth_bp = Blueprint("auth_bp", __name__)

@auth_bp.route('/api/login', methods=['POST'])
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