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
from code_evaluator import CodeQualityEvaluator
from ollama_client import OllamaClient

# -----------------------------
# Base Prompt Optimizer Class
# -----------------------------
class BaseCodePromptOptimizer(ABC):
    """Abstract base class for code prompt optimization methods"""
    
    def __init__(self, ollama_client: OllamaClient, dataset: List[Dict], name: str):
        self.ollama_client = ollama_client
        self.dataset = dataset
        self.name = name
        self.fitness_history = []
        self.diversity_history = []
        self.evaluator = CodeQualityEvaluator()
        
    @abstractmethod
    def optimize(self, generations: int = 5) -> Tuple[str, float]:
        """Main optimization method - returns best prompt and its score"""
        pass
    
    def evaluate_prompt(self, prompt: str) -> float:
        """Evaluate a single prompt against the code dataset"""
        total_score = 0
        total_items = len(self.dataset)
        
        for item in self.dataset:
            task_description = item["task"]
            test_cases = item.get("test_cases", [])
            
            # Generate code using the prompt
            full_prompt = f"{prompt}\n\nTask: {task_description}\n\nSolution:"
            
            try:
                generated_code = self.ollama_client.generate(full_prompt, max_tokens=400, temperature=0.1)
                
                if not generated_code:
                    continue
                
                # Extract code block
                code = self.evaluator.extract_code_block(generated_code)
                
                # Evaluate code quality
                syntax_valid, _ = self.evaluator.syntax_check(code)
                functional_score = self.evaluator.functional_test(code, test_cases)
                quality_metrics = self.evaluator.code_quality_metrics(code)
                
                # Calculate weighted score
                item_score = (
                    (1.0 if syntax_valid else 0.0) * 0.25 +  # Syntax
                    functional_score * 0.35 +  # Functionality
                    quality_metrics['readability'] * 0.15 +  # Readability
                    quality_metrics['complexity'] * 0.10 +  # Complexity
                    quality_metrics['documentation'] * 0.05 +  # Documentation
                    quality_metrics['best_practices'] * 0.10  # Best practices
                )
                
                total_score += item_score
                
            except Exception as e:
                print(f"Error evaluating prompt: {e}")
                continue
                
        return total_score / total_items if total_items > 0 else 0.0
    
    def calculate_diversity(self, population: List[str]) -> float:
        """Calculate diversity of prompt population"""
        if not population:
            return 0.0
            
        all_tokens = set()
        individual_tokens = []
        
        for prompt in population:
            tokens = set(prompt.lower().split())
            individual_tokens.append(tokens)
            all_tokens.update(tokens)
        
        if not all_tokens:
            return 0.0
            
        # Calculate average Jaccard similarity
        similarities = []
        for i in range(len(individual_tokens)):
            for j in range(i+1, len(individual_tokens)):
                intersection = len(individual_tokens[i].intersection(individual_tokens[j]))
                union = len(individual_tokens[i].union(individual_tokens[j]))
                if union > 0:
                    similarities.append(intersection / union)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        return 1 - avg_similarity
