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
import numpy as np
from base_optimizer import BaseCodePromptOptimizer
from typing import Tuple, Dict, List

class PromptOIRL(BaseCodePromptOptimizer):
    """Prompt Optimization via Iterative Reinforcement Learning"""
    
    def __init__(self, ollama_client, dataset: List[Dict]):
        super().__init__(ollama_client, dataset, "Prompt-OIRL")
        self.state_space = ["poor", "fair", "good", "excellent"]
        self.action_space = [
            "add_documentation", "simplify_code", "add_error_handling",
            "improve_naming", "optimize_performance", "enhance_readability"
        ]
        self.q_table = np.zeros((len(self.state_space), len(self.action_space)))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.3
    
    def get_state(self, score: float) -> int:
        """Map score to state index"""
        if score < 0.3:
            return 0  # poor
        elif score < 0.6:
            return 1  # fair
        elif score < 0.8:
            return 2  # good
        else:
            return 3  # excellent
    
    def apply_action(self, prompt: str, action: str) -> str:
        """Apply reinforcement learning action to prompt"""
        action_map = {
            "add_documentation": "Include comprehensive documentation and comments.",
            "simplify_code": "Write simple and straightforward code.",
            "add_error_handling": "Implement proper error handling and validation.",
            "improve_naming": "Use descriptive and meaningful variable names.",
            "optimize_performance": "Optimize for performance and efficiency.",
            "enhance_readability": "Focus on code readability and formatting."
        }
        return f"{prompt} {action_map[action]}"
    
    def optimize(self, generations: int = 8) -> Tuple[str, float]:
        """OIRL optimization"""
        current_prompt = "Write high-quality Python code."
        current_score = self.evaluate_prompt(current_prompt)
        current_state = self.get_state(current_score)
        
        best_prompt = current_prompt
        best_score = current_score
        
        for gen in range(generations):
            print(f"\n--- {self.name} Generation {gen+1}/{generations} ---")
            
            # Choose action (exploration vs exploitation)
            if np.random.random() < self.exploration_rate:
                action_idx = np.random.randint(len(self.action_space))
            else:
                action_idx = np.argmax(self.q_table[current_state])
            
            action = self.action_space[action_idx]
            new_prompt = self.apply_action(current_prompt, action)
            new_score = self.evaluate_prompt(new_prompt)
            new_state = self.get_state(new_score)
            
            # Update Q-table
            reward = new_score - current_score
            self.q_table[current_state, action_idx] += self.learning_rate * (
                reward + self.discount_factor * np.max(self.q_table[new_state]) - 
                self.q_table[current_state, action_idx]
            )
            
            if new_score > best_score:
                best_prompt, best_score = new_prompt, new_score
            
            current_prompt, current_score, current_state = new_prompt, new_score, new_state
            self.fitness_history.append(best_score)
            
            print(f"Action: {action}, Reward: {reward:.3f}, Score: {new_score:.3f}")
            print(f"Current prompt: {current_prompt}")
        
        return best_prompt, best_score