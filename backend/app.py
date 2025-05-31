# backend/app.py (UPDATED)
from flask import Flask
from flask_cors import CORS
import asyncio
from api import complexity, dsa_tracker, contest, health, algorithm_solver
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Enable async support
def make_async(f):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

# Register blueprints
app.register_blueprint(complexity.bp, url_prefix='/api/complexity')
app.register_blueprint(dsa_tracker.bp, url_prefix='/api/dsa')
app.register_blueprint(contest.bp, url_prefix='/api/contest')
app.register_blueprint(health.bp, url_prefix='/api/health')
app.register_blueprint(algorithm_solver.bp, url_prefix='/api/algorithm-solver')

@app.errorhandler(404)
def not_found(error):
    return {'error': 'Endpoint not found'}, 404

@app.errorhandler(500) 
def internal_error(error):
    return {'error': 'Internal server error'}, 500

if __name__ == '__main__':
    print("🤖 Starting Agentic AI Backend...")
    print("🧠 AI Agents: ComplexityAnalyzer, DSATracker, ContestOptimizer, AlgorithmSolver")
    print("⚡ Mode: Autonomous Agent Processing")
    print("📚 Enhanced RAG: MongoDB + FAISS + Reinforcement Learning")
    app.run(host='0.0.0.0', port=5000, debug=True)
