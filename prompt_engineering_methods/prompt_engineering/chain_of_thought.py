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

# -----------------------------
# Chain-of-Thought for Code
# -----------------------------
class ChainOfThoughtCode(BaseCodePromptOptimizer):
    """Chain-of-Thought prompting for code generation"""
    
    def __init__(self, ollama_client: OllamaClient, dataset: List[Dict]):
        super().__init__(ollama_client, dataset, "Chain-of-Thought-Code")
    
    def optimize(self, generations: int = 5) -> Tuple[str, float]:
        """CoT optimization for code generation"""
        
        cot_prompts = [
            "Let's think step by step to write clean Python code: First analyze the problem, then design the solution, finally implement with proper structure and testing.",
            "Step-by-step approach: 1) Understand requirements 2) Plan the algorithm 3) Write clean, documented code 4) Test functionality.",
            "Breaking this down: analyze the task, identify key components, implement efficient solution, ensure code quality.",
            "Systematic approach: examine problem → design algorithm → code implementation → verify correctness.",
            "Think through this systematically: problem analysis, solution design, clean implementation, proper testing."
        ]
        
        best_prompt = ""
        best_score = 0
        
        for gen, prompt in enumerate(cot_prompts):
            print(f"\n--- {self.name} Generation {gen+1}/{len(cot_prompts)} ---")
            
            score = self.evaluate_prompt(prompt)
            self.fitness_history.append(score)
            self.diversity_history.append(1.0)
            
            print(f"Score: {score:.3f}")
            print(f"Prompt: {prompt[:80]}...")
            
            if score > best_score:
                best_score = score
                best_prompt = prompt
        
        return best_prompt, best_score
