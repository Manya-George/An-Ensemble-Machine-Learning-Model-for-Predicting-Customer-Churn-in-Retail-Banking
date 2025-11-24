from app_routes.auth.login_routes import auth_bp
from app_routes.auth.auth_routes import userauth_bp
from app_routes.auth.otp_routes import otp_bp
from app_routes.auth.logout_routes import endsession_bp
from app_routes.predictions.upload_routes import upload_bp
from app_routes.predictions.results_routes import results_bp
from app_routes.predictions.request_routes import requests_bp
from app_routes.insights.insight_routes import insights_bp
from app_routes.insights.analytics_routes import analytics_bp
from app_routes.user_management.crud_routes import crud_bp
from app_routes.user_management.dashboard_routes import dashboard_bp
from app_routes.logs.logs_routes import logs_bp

def register_blueprints(app):
    app.register_blueprint(auth_bp)
    app.register_blueprint(userauth_bp)
    app.register_blueprint(otp_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(results_bp)
    app.register_blueprint(requests_bp)
    app.register_blueprint(insights_bp)
    app.register_blueprint(analytics_bp)    
    app.register_blueprint(crud_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(logs_bp)