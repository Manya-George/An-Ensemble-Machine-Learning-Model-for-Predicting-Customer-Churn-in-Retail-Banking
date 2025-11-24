from flask import Blueprint, request, jsonify, session
from core.utils import login_required, log_system_event
import bcrypt
from core.helpers import get_db_connection

crud_bp = Blueprint("crud_bp", __name__)

@crud_bp.route('/api/register-user', methods=['POST'])
def register_user():
    """Register a new user in the LoyaltyLens system"""
    try:
        data = request.json
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        role = data.get('role')

        if not username or not email or not password or not role:
            return jsonify({'success': False, 'message': 'All fields are required'}), 400

        if len(password) < 8:
            return jsonify({'success': False, 'message': 'Password must be at least 8 characters long'}), 400

        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT userID FROM users WHERE username=%s OR email=%s", (username, email))
        existing_user = cursor.fetchone()
        if existing_user:
            cursor.close()
            conn.close()
            try:
                log_system_event("/api/register-user", "POST", f"Registration failed - duplicate {username} or {email}", 409, user_id=session.get('user_id'))
            except:
                pass
            return jsonify({'success': False, 'message': 'Username or email already exists'}), 409

        cursor.execute(
            """INSERT INTO users (username, email, password_hash, role)
               VALUES (%s, %s, %s, %s)""",
            (username, email, password_hash, role)
        )
        conn.commit()
        cursor.close()
        conn.close()

        try:
            log_system_event("/api/register-user", "POST", f"User registered: {username}", 201, user_id=session.get('user_id'))
        except:
            pass

        return jsonify({
            'success': True,
            'message': f'User \"{username}\" registered successfully!'
        }), 201

    except Exception as e:
        print(f"Registration error: {e}")
        traceback.print_exc()
        try:
            log_system_event("/api/register-user", "POST", f"Registration error: {e}", 500, user_id=session.get('user_id'))
        except:
            pass
        return jsonify({'success': False, 'message': 'Server error'}), 500

@crud_bp.route('/api/lookup-user', methods=['POST'])
def lookup_user():
    try:
        data = request.json
        username = data.get('username')

        if not username:
            return jsonify({'success': False, 'message': 'Username is required'}), 400

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT username, email, role FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if not user:
            return jsonify({'success': False, 'message': 'User not found'}), 404

        return jsonify({'success': True, 'user': user}), 200

    except Exception as e:
        print(f"Lookup user error: {e}")
        return jsonify({'success': False, 'message': 'Server error'}), 500

@crud_bp.route('/api/update-user', methods=['PUT'])
def update_user():
    try:
        data = request.json
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        role = data.get('role')

        if not username or not email or not role:
            return jsonify({'success': False, 'message': 'Missing required fields'}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        if password:
            import bcrypt
            hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            cursor.execute("""
                UPDATE users 
                SET email=%s, password_hash=%s, role=%s, updatedAt=NOW() 
                WHERE username=%s
            """, (email, hashed, role, username))
        else:
            cursor.execute("""
                UPDATE users 
                SET email=%s, role=%s, updatedAt=NOW() 
                WHERE username=%s
            """, (email, role, username))

        affected = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()

        if affected == 0:
            return jsonify({'success': False, 'message': 'User not found or no changes made'}), 404

        return jsonify({'success': True, 'message': 'User updated successfully'}), 200

    except Exception as e:
        print(f"Update user error: {e}")
        return jsonify({'success': False, 'message': 'Server error'}), 500

@crud_bp.route('/api/revoke-user', methods=['DELETE'])
def revoke_user():
    try:
        data = request.json
        username = data.get('username')

        if not username:
            return jsonify({'success': False, 'message': 'Username is required'}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        # Soft revoke (avoid deletion to keep FK integrity)
        cursor.execute("""
            UPDATE users 
            SET role = 'Revoked', is_active = FALSE, updatedAt = NOW() 
            WHERE username = %s
        """, (username,))

        conn.commit()
        affected = cursor.rowcount
        cursor.close()
        conn.close()

        if affected == 0:
            return jsonify({'success': False, 'message': 'User not found'}), 404

        return jsonify({'success': True, 'message': 'User access revoked successfully'}), 200

    except Exception as e:
        print(f"Revoke user error: {e}")
        return jsonify({'success': False, 'message': 'Server error'}), 500