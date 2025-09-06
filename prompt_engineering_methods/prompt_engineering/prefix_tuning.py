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
from base_optimizer import BaseCodePromptOptimizer
from ollama_client import OllamaClient
from base_optimizer import BaseCodePromptOptimizer
from typing import Tuple

class PrefixTuning(BaseCodePromptOptimizer):
    """Prefix Tuning for code generation"""
    
    def __init__(self, ollama_client, dataset: List[Dict]):
        super().__init__(ollama_client, dataset, "Prefix-Tuning")
        self.prefixes = [
            "As an expert Python developer, ",
            "Following best practices, ",
            "Using clean code principles, ",
            "With proper error handling, ",
            "Applying software engineering patterns, "
        ]
    
    def optimize(self, generations: int = 5) -> Tuple[str, float]:
        """Prefix tuning optimization"""
        base_instruction = "write high-quality Python code for the following task:"
        
        best_prefix = ""
        best_score = 0
        
        for gen in range(generations):
            print(f"\n--- {self.name} Generation {gen+1}/{generations} ---")
            
            # Test all prefixes
            scores = []
            for prefix in self.prefixes:
                full_prompt = f"{prefix}{base_instruction}"
                score = self.evaluate_prompt(full_prompt)
                scores.append((prefix, score))
                print(f"Prefix: {prefix} -> Score: {score:.3f}")
            
            # Find best prefix
            scores.sort(key=lambda x: x[1], reverse=True)
            best_current_prefix, best_current_score = scores[0]
            
            if best_current_score > best_score:
                best_prefix, best_score = best_current_prefix, best_current_score
            
            self.fitness_history.append(best_score)
        
        best_prompt = f"{best_prefix}{base_instruction}"
        return best_prompt, best_score