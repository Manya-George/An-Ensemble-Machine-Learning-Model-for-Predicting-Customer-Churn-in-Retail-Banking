from flask import Flask
from flask_session import Session
from flask_cors import CORS
from core.config import SESSION_CONFIG
from app_routes import register_blueprints

app = Flask(__name__)
app.config.update(SESSION_CONFIG)

Session(app)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

register_blueprints(app)

if __name__ == "__main__":
    app.run(debug=True)
