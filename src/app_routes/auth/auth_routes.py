from flask import Blueprint, jsonify, session

userauth_bp = Blueprint("userauth_bp", __name__)

@userauth_bp.route('/api/check-auth', methods=['GET'])
def check_auth():
    """Check if user is authenticated"""
    if session.get('authenticated'):
        return jsonify({
            'authenticated': True,
            'username': session.get('username'),
            'role': session.get('role')
        }), 200
    return jsonify({'authenticated': False}), 401