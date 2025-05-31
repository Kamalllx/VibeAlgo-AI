
# backend/api/dsa_tracker.py (FIXED)
from flask import Blueprint, request, jsonify
import asyncio
from core.agent_orchestrator import orchestrator
from core.dsa_progress_tracker import update_progress, get_progress

bp = Blueprint('dsa_tracker', __name__)

@bp.route('/progress/<user_id>', methods=['GET'])
def get_user_progress(user_id):
    """Get DSA progress - FIXED VERSION"""
    try:
        print(f"\n📊 API: Getting progress for user {user_id}")
        
        # Use the working progress tracker
        progress_data = get_progress(user_id)
        
        print(f"✅ Retrieved progress data: {len(progress_data.get('problems', []))} problems")
        
        return jsonify({
            'success': True,
            'data': progress_data
        })
        
    except Exception as e:
        print(f"❌ Progress retrieval error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/progress/<user_id>/update', methods=['POST'])
def update_user_progress(user_id):
    """Update progress with optional AI analysis"""
    try:
        data = request.get_json()
        print(f"\n📝 API: Updating progress for {user_id}")
        print(f"📊 Problem: {data.get('problem_id')} - {data.get('status')}")
        
        # Update progress using existing tracker
        result = update_progress(
            user_id=user_id,
            problem_id=data.get('problem_id'),
            status=data.get('status'),
            difficulty=data.get('difficulty', 'medium'),
            topic=data.get('topic', 'general'),
            time_spent=data.get('time_spent', 0)
        )
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        print(f"❌ Progress update error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
