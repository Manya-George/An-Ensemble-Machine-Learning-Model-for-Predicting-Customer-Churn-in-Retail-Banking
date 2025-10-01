from flask import Flask, request, jsonify, session
from flask_cors import CORS
import mysql.connector
import bcrypt
import random
import string
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()
print(f"Email configured: {os.getenv('EMAIL_ADDRESS')}")
print(f"Email password exists: {bool(os.getenv('EMAIL_PASSWORD'))}")

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
CORS(app, supports_credentials=True)


# Database connection
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS")
)

# Email configuration
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email': os.getenv('EMAIL_ADDRESS', 'your-email@gmail.com'),
    'password': os.getenv('EMAIL_PASSWORD', 'your-app-password')
}

def get_db_connection():
    """Create database connection"""
    return mysql.connector.connect(**DB_CONFIG)

def generate_otp():
    """Generate a 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=6))

def send_otp_email(email, otp_code, username):
    """Send OTP via email"""
    try:
        # Debug output
        print(f"Attempting to send email to: {email}")
        print(f"Using email account: {EMAIL_CONFIG['email']}")
        print(f"SMTP Server: {EMAIL_CONFIG['smtp_server']}:{EMAIL_CONFIG['smtp_port']}")
        
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
        
        print("Connecting to SMTP server...")
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.set_debuglevel(1)  # Enable debug output
        
        print("Starting TLS...")
        server.starttls()
        
        print("Logging in...")
        server.login(EMAIL_CONFIG['email'], EMAIL_CONFIG['password'])
        
        print("Sending message...")
        server.send_message(msg)
        
        print("Closing connection...")
        server.quit()
        
        print("✅ Email sent successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/api/login', methods=['POST'])
def login():
    """Handle user login"""
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get user from database
        cursor.execute(
            "SELECT userID, username, email, password_hash, role FROM users WHERE username = %s",
            (username,)
        )
        user = cursor.fetchone()
        
        if not user:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401

        # ADD THIS DEBUG CODE:
        print(f"User found: {user['username']}")
        print(f"Stored hash: {user['password_hash']}")
        print(f"Password entered: {password}")
        
        # Verify password
        if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
        
        # Generate OTP
        otp_code = generate_otp()
        expires_at = datetime.now() + timedelta(minutes=1)
        
        # Store OTP in database
        cursor.execute(
            "INSERT INTO otps (userID, otp_code, expiresAt) VALUES (%s, %s, %s)",
            (user['userID'], otp_code, expires_at)
        )
        conn.commit()
        
        # Mask email for display
        email_parts = user['email'].split('@')
        masked_email = f"{email_parts[0][0]}{'*' * (len(email_parts[0]) - 1)}@{email_parts[1]}"
        
        # Send OTP via email
        email_sent = send_otp_email(user['email'], otp_code, user['username'])
        
        if not email_sent:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Failed to send OTP email'}), 500
        
        # Store user info in session
        session['temp_user_id'] = user['userID']
        session['username'] = user['username']
        session['role'] = user['role']
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'OTP sent successfully',
            'masked_email': masked_email
        }), 200
        
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'success': False, 'message': 'Server error'}), 500

@app.route('/api/verify-otp', methods=['POST'])
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
        
        # Get the latest unused OTP for this user
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
        
        # Check if OTP has expired
        if datetime.now() > otp_record['expiresAt']:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'OTP has expired'}), 401
        
        # Verify OTP code
        if otp_code != otp_record['otp_code']:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Invalid OTP'}), 401
        
        # Mark OTP as used
        cursor.execute(
            "UPDATE otps SET isUsed = TRUE WHERE otpID = %s",
            (otp_record['otpID'],)
        )
        conn.commit()
        
        # Set authenticated session
        session['user_id'] = user_id
        session['authenticated'] = True
        session.pop('temp_user_id', None)
        
        role = session.get('role')
        username = session.get('username')
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'OTP verified successfully',
            'role': role,
            'username': username
        }), 200
        
    except Exception as e:
        print(f"OTP verification error: {e}")
        return jsonify({'success': False, 'message': 'Server error'}), 500

@app.route('/api/resend-otp', methods=['POST'])
def resend_otp():
    """Resend OTP to user"""
    try:
        user_id = session.get('temp_user_id')
        if not user_id:
            return jsonify({'success': False, 'message': 'Session expired. Please login again'}), 401
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get user info
        cursor.execute(
            "SELECT username, email FROM users WHERE userID = %s",
            (user_id,)
        )
        user = cursor.fetchone()
        
        if not user:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'User not found'}), 404
        
        # Generate new OTP
        otp_code = generate_otp()
        expires_at = datetime.now() + timedelta(minutes=1)
        
        # Store new OTP
        cursor.execute(
            "INSERT INTO otps (userID, otp_code, expiresAt) VALUES (%s, %s, %s)",
            (user_id, otp_code, expires_at)
        )
        conn.commit()
        
        # Send OTP via email
        email_sent = send_otp_email(user['email'], otp_code, user['username'])
        
        cursor.close()
        conn.close()
        
        if not email_sent:
            return jsonify({'success': False, 'message': 'Failed to send OTP email'}), 500
        
        return jsonify({
            'success': True,
            'message': 'OTP resent successfully'
        }), 200
        
    except Exception as e:
        print(f"Resend OTP error: {e}")
        return jsonify({'success': False, 'message': 'Server error'}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout user"""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'}), 200

@app.route('/api/check-auth', methods=['GET'])
def check_auth():
    """Check if user is authenticated"""
    if session.get('authenticated'):
        return jsonify({
            'authenticated': True,
            'username': session.get('username'),
            'role': session.get('role')
        }), 200
    return jsonify({'authenticated': False}), 401

if __name__ == '__main__':
    app.run(debug=True, port=5000)