import os
from datetime import timedelta

class Config:
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(24))
    ENV = os.environ.get('FLASK_ENV', 'production')
    DEBUG = os.environ.get('FLASK_DEBUG', '0') == '1'
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///site.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(days=1)
    
    # Rate Limiting
    RATE_LIMIT_WINDOW = 1800  # 30 minutes in seconds
    RATE_LIMIT_MAX_REQUESTS = 5
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'logs/app.log')