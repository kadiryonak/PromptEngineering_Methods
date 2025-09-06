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
# AutoPrompt for Code Generation
# -----------------------------
class AutoPromptCodeGA(BaseCodePromptOptimizer):
    """Genetic Algorithm-based AutoPrompt for code generation"""
    
    def __init__(self, ollama_client: OllamaClient, dataset: List[Dict], selection_method: str = "tournament"):
        super().__init__(ollama_client, dataset, f"AutoPrompt-Code-{selection_method}")
        self.selection_method = selection_method
        self.population_size = 8
        
        # Code-specific word bank
        self.word_bank = [
            "implement", "create", "develop", "build", "design",
            "function", "method", "algorithm", "solution", "code",
            "efficient", "clean", "readable", "maintainable", "robust",
            "python", "programming", "software", "logic", "structure",
            "optimize", "debug", "test", "validate", "handle",
            "properly", "correctly", "accurately", "systematically",
            "step-by-step", "modular", "reusable", "scalable"
        ]
    
    def mutate_prompt(self, prompt: str) -> str:
        """Code-optimized mutation"""
        words = prompt.split()
        
        if not words:
            return random.choice([
                "Write clean and efficient Python code",
                "Implement a well-structured solution",
                "Create readable and maintainable code"
            ])
        
        mutation_type = random.choice(["add", "remove", "replace", "reorder"])
        
        if mutation_type == "add" and len(words) < 30:
            new_word = random.choice(self.word_bank)
            position = random.randint(0, len(words))
            words.insert(position, new_word)
            
        elif mutation_type == "remove" and len(words) > 3:
            words.pop(random.randint(0, len(words) - 1))
            
        elif mutation_type == "replace" and words:
            position = random.randint(0, len(words) - 1)
            words[position] = random.choice(self.word_bank)
            
        elif mutation_type == "reorder" and len(words) > 3:
            start = random.randint(0, len(words) - 3)
            end = min(start + 3, len(words))
            segment = words[start:end]
            random.shuffle(segment)
            words[start:end] = segment
        
        return " ".join(words)
    
    def crossover(self, parent1: str, parent2: str) -> str:
        """Smart crossover for code prompts"""
        words1, words2 = parent1.split(), parent2.split()
        
        if not words1:
            return parent2
        if not words2:
            return parent1
            
        min_len = min(len(words1), len(words2))
        if min_len <= 1:
            return parent1 if len(words1) > len(words2) else parent2
            
        cut_point = random.randint(1, min_len - 1)
        
        if random.random() < 0.5:
            child_words = words1[:cut_point] + words2[cut_point:]
        else:
            child_words = words2[:cut_point] + words1[cut_point:]
        
        return " ".join(child_words)
    
    def tournament_selection(self, scored_population: List[Tuple[str, float]]) -> str:
        """Tournament selection"""
        tournament = random.sample(scored_population, min(3, len(scored_population)))
        return max(tournament, key=lambda x: x[1])[0]
    
    def optimize(self, generations: int = 8) -> Tuple[str, float]:
        """Code-optimized genetic algorithm"""
        
        # Initial population for code generation
        population = [
            "Write clean, efficient Python code with proper structure and documentation.",
            "Implement a well-designed solution using best programming practices.",
            "Create readable and maintainable code that follows Python conventions.",
            "Develop robust code with proper error handling and clear logic.",
            "Build an optimized solution with good performance and readability.",
            "Design modular and reusable code with clear function definitions.",
            "Implement systematic programming approach with proper testing.",
            "Create professional-quality code with comprehensive functionality."
        ]
        
        for gen in range(generations):
            print(f"\n--- {self.name} Generation {gen+1}/{generations} ---")
            
            # Evaluate population
            scored = [(prompt, self.evaluate_prompt(prompt)) for prompt in population]
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # Track metrics
            current_best = scored[0][1]
            self.fitness_history.append(current_best)
            self.diversity_history.append(self.calculate_diversity(population))
            
            print(f"Best score: {current_best:.3f} | Diversity: {self.diversity_history[-1]:.3f}")
            print(f"Best prompt: {scored[0][0][:80]}...")
            
            # Create new population
            new_population = []
            
            # Elite preservation
            survivors = max(2, self.population_size // 4)
            for i in range(survivors):
                new_population.append(scored[i][0])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(scored)
                parent2 = self.tournament_selection(scored)
                
                child = self.crossover(parent1, parent2)
                
                if random.random() < 0.7:
                    child = self.mutate_prompt(child)
                
                new_population.append(child)
            
            population = new_population[:self.population_size]
        
        # Return best result
        final_scored = [(prompt, self.evaluate_prompt(prompt)) for prompt in population]
        best_prompt, best_score = max(final_scored, key=lambda x: x[1])
        
        return best_prompt, best_score
