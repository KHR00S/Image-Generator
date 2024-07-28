from flask import Flask
from .app import generate

def create_app():
    app = Flask(__name__)

    # Register blueprints or routes
    app.add_url_rule('/generate', 'generate', generate, methods=['POST'])

    return app
