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
# Few-Shot Learning for Code
# -----------------------------
class FewShotCode(BaseCodePromptOptimizer):
    """Few-shot learning for code generation"""
    
    def __init__(self, ollama_client: OllamaClient, dataset: List[Dict]):
        super().__init__(ollama_client, dataset, "Few-Shot-Code")
    
    def optimize(self, generations: int = 4) -> Tuple[str, float]:
        """Few-shot optimization"""
        
        few_shot_prompts = [
            """Here are examples of good Python code:

Example 1:
def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

Example 2:
def find_maximum(lst):
    if not lst:
        return None
    max_val = lst[0]
    for num in lst[1:]:
        if num > max_val:
            max_val = num
    return max_val

Now write similar clean, efficient Python code:""",
            
            """Examples of well-structured Python functions:

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def is_palindrome(text):
    text = text.lower().replace(' ', '')
    return text == text[::-1]

Following these patterns, create clean Python code:""",
            
            """Good Python code examples:

def sort_list(items):
    return sorted(items)

def count_words(text):
    return len(text.split())

Write similar quality code:""",
            
            """Examples:
def reverse_string(s): return s[::-1]
def is_even(n): return n % 2 == 0

Create equivalent quality code:"""
        ]
        
        best_prompt = ""
        best_score = 0
        
        for gen, prompt in enumerate(few_shot_prompts):
            print(f"\n--- {self.name} Generation {gen+1}/{len(few_shot_prompts)} ---")
            
            score = self.evaluate_prompt(prompt)
            self.fitness_history.append(score)
            self.diversity_history.append(1.0)
            
            print(f"Score: {score:.3f}")
            print(f"Prompt type: Few-shot example {gen+1}")
            
            if score > best_score:
                best_score = score
                best_prompt = prompt
        
        return best_prompt, best_score