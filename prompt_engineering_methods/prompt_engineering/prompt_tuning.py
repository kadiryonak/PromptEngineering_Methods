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
import random
from base_optimizer import BaseCodePromptOptimizer
from typing import Tuple, List

class PromptTuning(BaseCodePromptOptimizer):
    """Prompt Tuning for code generation"""
    
    def __init__(self, ollama_client, dataset: List[Dict]):
        super().__init__(ollama_client, dataset, "Prompt-Tuning")
        self.soft_prompts = []
        self.tuning_words = [
            "efficient", "clean", "readable", "maintainable", "robust",
            "well-structured", "documented", "tested", "optimized", "modular"
        ]
    
    def optimize(self, generations: int = 6) -> Tuple[str, float]:
        """Prompt tuning optimization"""
        base_prompt = "Write Python code that is"
        
        best_prompt = base_prompt
        best_score = self.evaluate_prompt(best_prompt)
        
        for gen in range(generations):
            print(f"\n--- {self.name} Generation {gen+1}/{generations} ---")
            
            # Generate candidate prompts
            candidates = []
            for _ in range(5):
                num_adjectives = random.randint(2, 4)
                adjectives = random.sample(self.tuning_words, num_adjectives)
                candidate = f"{base_prompt} {', '.join(adjectives)}."
                candidates.append((candidate, self.evaluate_prompt(candidate)))
            
            # Select best candidate
            candidates.sort(key=lambda x: x[1], reverse=True)
            current_best, current_score = candidates[0]
            
            if current_score > best_score:
                best_prompt, best_score = current_best, current_score
                print(f"✓ Improved to: {best_score:.3f}")
            else:
                print(f"✗ No improvement: {current_score:.3f}")
            
            print(f"Best prompt: {best_prompt}")
            self.fitness_history.append(best_score)
        
        return best_prompt, best_score