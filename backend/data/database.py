# backend/data/database.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///app.db')

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserProgress(Base):
    __tablename__ = "user_progress"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    problem_id = Column(String, index=True)
    status = Column(String)
    difficulty = Column(String)
    topic = Column(String)
    attempts = Column(Integer, default=1)
    time_spent = Column(Integer, default=0)
    solution_efficiency = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ComplexityAnalysis(Base):
    __tablename__ = "complexity_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    code_hash = Column(String, index=True)
    language = Column(String)
    platform = Column(String)
    time_complexity = Column(String)
    space_complexity = Column(String)
    analysis_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class ContestSubmission(Base):
    __tablename__ = "contest_submissions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    contest_id = Column(String, index=True)
    problem_id = Column(String)
    submission_time = Column(DateTime)
    time_taken = Column(Integer)
    status = Column(String)
    score = Column(Float)
    optimization_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

def create_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
