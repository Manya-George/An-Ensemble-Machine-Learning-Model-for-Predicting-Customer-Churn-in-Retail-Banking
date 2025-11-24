from flask import Blueprint, jsonify, session, send_from_directory, current_app
import os
from core.utils import login_required, log_system_event

analytics_bp = Blueprint("analytics_bp", __name__)

@analytics_bp.route('/api/interpretability-data', methods=['GET'])
def get_interpretability_data():
    """Serves the structured JSON output containing P-values and SHAP values."""
    try:
        # Correct way to reference files in a blueprint
        JSON_DIR = os.path.join(current_app.root_path, "analytics_outputs")
        JSON_FILENAME = "interpretability_dashboard_data.json"
        JSON_PATH = os.path.join(JSON_DIR, JSON_FILENAME)

        if not os.path.exists(JSON_PATH):
            return jsonify({
                "success": False,
                "message": "Interpretability data not found. Please run model training first."
            }), 404

        return send_from_directory(JSON_DIR, JSON_FILENAME)

    except Exception as e:
        print(f"Error fetching interpretability data: {e}")
        log_system_event(
            path="/api/interpretability-data",
            method="GET",
            description=f"Server error: {str(e)}",
            status_code=500,
            user_id=session.get("user_id")
        )
        return jsonify({'success': False, 'message': 'Server error reading analytics file.'}), 500


@analytics_bp.route('/api/analytics-plots/<filename>', methods=['GET'])
def serve_analytics_plot(filename):
    """Serves the individual PNG plot files."""
    try:
        PLOT_DIR = os.path.join(current_app.root_path, "analytics_outputs", "plots")
        return send_from_directory(PLOT_DIR, filename)

    except FileNotFoundError:
        return jsonify({'success': False, 'message': f'Plot file {filename} not found.'}), 404

    except Exception as e:
        print(f"Error serving plot {filename}: {e}")
        log_system_event(
            path=f"/api/analytics-plots/{filename}",
            method="GET",
            description=f"Server error: {str(e)}",
            status_code=500,
            user_id=session.get("user_id")
        )
        return jsonify({'success': False, 'message': 'Error serving plot'}), 500
