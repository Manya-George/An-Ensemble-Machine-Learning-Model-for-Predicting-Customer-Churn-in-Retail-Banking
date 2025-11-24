from flask import Blueprint, request, jsonify, session
import traceback
from core.utils import log_system_event

endsession_bp = Blueprint("endsession_bp", __name__)

@endsession_bp.route('/api/logout', methods=['POST'])
def logout():
    """Logout user"""
    try:
        user_id = session.get('user_id')
        session.clear()
        try:
            log_system_event("/api/logout", "POST", f"User {user_id} logged out", 200, user_id=user_id)
        except:
            pass
        return jsonify({'success': True, 'message': 'Logged out successfully'}), 200
    except Exception as e:
        print("Logout error:", e)
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Logout failed'}), 500