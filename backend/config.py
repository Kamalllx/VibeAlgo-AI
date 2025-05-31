# backend/config.py (Enhanced)
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DEBUG = os.environ.get('FLASK_ENV') == 'development'
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    GROQ_API_KEY ="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"
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'chrome-extension://*').split(',')
    
    # AI Configuration
    MAX_CODE_LENGTH = 10000
    DEFAULT_MODEL = 'llama3-70b-8192'
    
    # Performance Configuration  
    REQUEST_TIMEOUT = 30
    MAX_WORKERS = 4

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}