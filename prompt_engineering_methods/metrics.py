import random
import numpy as np
import requests
import json
from abc import ABC, abstractmethod
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from collections import defaultdict
import re
import ast

# -----------------------------
# Evaluation Metrics Class
# -----------------------------
@dataclass
class CodeEvaluationMetrics:
    """Comprehensive evaluation metrics for code prompt optimization"""
    
    syntax_score: float = 0.0
    functional_score: float = 0.0
    readability_score: float = 0.0
    complexity_score: float = 0.0
    documentation_score: float = 0.0
    best_practices_score: float = 0.0
    overall_score: float = 0.0
    convergence_speed: float = 0.0
    diversity_maintained: float = 0.0
    
    def to_dict(self):
        return {
            'syntax_score': self.syntax_score,
            'functional_score': self.functional_score,
            'readability_score': self.readability_score,
            'complexity_score': self.complexity_score,
            'documentation_score': self.documentation_score,
            'best_practices_score': self.best_practices_score,
            'overall_score': self.overall_score,
            'convergence_speed': self.convergence_speed,
            'diversity_maintained': self.diversity_maintained
        }
