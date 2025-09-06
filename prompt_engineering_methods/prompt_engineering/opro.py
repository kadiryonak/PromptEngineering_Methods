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
# OPRO for Code Generation
# -----------------------------
class OPROCode(BaseCodePromptOptimizer):
    """OPRO implementation for code generation"""
    
    def __init__(self, ollama_client: OllamaClient, dataset: List[Dict]):
        super().__init__(ollama_client, dataset, "OPRO-Code")
    
    def generate_improved_prompt(self, current_prompt: str, performance_feedback: str) -> str:
        """Generate improved code generation prompts"""
        
        improvement_prompt = f"""You are a prompt engineering expert. Your task is to improve code generation instructions.

Current instruction: "{current_prompt}"
Performance analysis: {performance_feedback}

Requirements for the improved instruction:
1. Focus on clean, readable Python code
2. Emphasize proper structure and best practices
3. Include guidance for error handling and testing
4. Optimize for maintainability and efficiency

Generate a better instruction for code generation (one sentence, max 50 words):"""
        
        try:
            improved_text = self.ollama_client.generate(improvement_prompt, max_tokens=100, temperature=0.7)
            
            if improved_text and len(improved_text.strip()) > 10:
                # Extract the actual improved prompt
                lines = improved_text.strip().split('\n')
                for line in lines:
                    clean_line = line.strip()
                    if len(clean_line) > 10 and not clean_line.startswith(('You are', 'Current', 'Performance', 'Requirements')):
                        return clean_line
                        
            return current_prompt  # Fallback
                
        except Exception as e:
            print(f"Error in OPRO generation: {e}")
            return current_prompt
    
    def optimize(self, generations: int = 6) -> Tuple[str, float]:
        """OPRO optimization for code generation"""
        
        current_prompt = "Write clean, efficient Python code with proper structure."
        
        for gen in range(generations):
            print(f"\n--- {self.name} Generation {gen+1}/{generations} ---")
            
            current_score = self.evaluate_prompt(current_prompt)
            self.fitness_history.append(current_score)
            self.diversity_history.append(1.0)
            
            print(f"Current score: {current_score:.3f}")
            print(f"Current prompt: {current_prompt}")
            
            # Generate performance feedback
            if current_score < 0.3:
                feedback = "Poor code quality - needs better structure and syntax guidance"
            elif current_score < 0.5:
                feedback = "Moderate quality - improve readability and functionality"
            elif current_score < 0.7:
                feedback = "Good quality - fine-tune for best practices and documentation"
            else:
                feedback = "High quality - optimize for edge cases and maintainability"
            
            # Generate improvement
            improved_prompt = self.generate_improved_prompt(current_prompt, feedback)
            improved_score = self.evaluate_prompt(improved_prompt)
            
            if improved_score > current_score:
                current_prompt = improved_prompt
                print(f"✓ Accepted: {improved_score:.3f} > {current_score:.3f}")
            else:
                print(f"✗ Rejected: {improved_score:.3f} <= {current_score:.3f}")
        
        return current_prompt, self.evaluate_prompt(current_prompt)
