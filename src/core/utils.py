import os
import random
import string
from flask import session, jsonify
from functools import wraps
from datetime import datetime
from core.helpers import get_db_connection
from core.config import EMAIL_CONFIG
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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

def log_system_event(path, method, desc, user_id=None):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO api_metrics (path, method, userID, timestamp)
            VALUES (%s, %s, %s, %s)
        """, (path, method, user_id, datetime.utcnow()))
        conn.commit()
        cursor.close()
        conn.close()
    except:
        pass

def calculate_risk_level(probability):
    """Determine risk level based on the dynamic optimal threshold."""
    if probability >= THRESHOLD:
        return 'high'
    elif probability >= (0.5 + (THRESHOLD - 0.5) * 0.75): 
        return 'medium'
    else:
        return 'low'

def login_required(fn):
    """
    Decorator to enforce authentication. Returns 401 if not authenticated.
    Use on routes that require login.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get('authenticated'):
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        return fn(*args, **kwargs)
    return wrapper


