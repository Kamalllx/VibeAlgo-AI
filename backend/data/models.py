
# backend/data/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ProblemStatus(str, Enum):
    ATTEMPTED = "attempted"
    SOLVED = "solved"
    MASTERED = "mastered"

class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class CodeAnalysisRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=10000)
    language: str = Field(default="python")
    platform: str = Field(default="general")
    user_id: Optional[str] = None

class ProgressUpdateRequest(BaseModel):
    problem_id: str = Field(..., min_length=1)
    status: ProblemStatus
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    topic: str = Field(default="general")
    time_spent: int = Field(default=0, ge=0)

class ContestOptimizationRequest(BaseModel):
    contest_id: str = Field(..., min_length=1)
    submissions: List[Dict[str, Any]]
    user_id: Optional[str] = None

class ComplexityResult(BaseModel):
    time_complexity: str
    space_complexity: str
    loop_analysis: Dict[str, Any]
    suggestions: List[str]
    visualization_data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
