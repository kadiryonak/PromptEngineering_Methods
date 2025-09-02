import random
from base_class import BasePromptOptimizer
# -----------------------------
# Genetic Algorithm Base Class
# -----------------------------
class GeneticAlgorithmBase(BasePromptOptimizer):
    """Base class for genetic algorithm-based prompt optimizers"""
    
    def __init__(self, model_pipeline, dataset: List[Dict], name: str, 
                 population_size: int = 8, selection_method: str = "tournament"):
        super().__init__(model_pipeline, dataset, name)
        self.population_size = population_size
        self.selection_method = selection_method
        self.tournament_size = 3
        
    # Selection Methods
    def tournament_selection(self, scored_population: List[Tuple[str, float]], tournament_size: int = None) -> Tuple[str, float]:
        """Tournament Selection: Select best individual from a random tournament"""
        if tournament_size is None:
            tournament_size = self.tournament_size
        tournament = random.sample(scored_population, min(tournament_size, len(scored_population)))
        return max(tournament, key=lambda x: x[1])
    
    def roulette_wheel_selection(self, scored_population: List[Tuple[str, float]]) -> Tuple[str, float]:
        """Roulette Wheel Selection: Probability-based selection proportional to fitness"""
        total_fitness = sum(score for _, score in scored_population)
        if total_fitness == 0:
            return random.choice(scored_population)
        
        # Handle negative fitness by shifting
        min_fitness = min(score for _, score in scored_population)
        if min_fitness < 0:
            adjusted_scores = [(prompt, score - min_fitness + 0.1) for prompt, score in scored_population]
        else:
            adjusted_scores = [(prompt, score + 0.1) for prompt, score in scored_population]  # Add small epsilon
        
        total_adjusted = sum(score for _, score in adjusted_scores)
        pick = random.uniform(0, total_adjusted)
        current = 0
        
        for prompt, score in adjusted_scores:
            current += score
            if current >= pick:
                return (prompt, score)
        
        return adjusted_scores[-1]
    
    def rank_selection(self, scored_population: List[Tuple[str, float]]) -> Tuple[str, float]:
        """Rank Selection: Selection based on rank rather than raw fitness values"""
        ranked = sorted(scored_population, key=lambda x: x[1])
        weights = list(range(1, len(ranked) + 1))
        total_weight = sum(weights)
        
        pick = random.uniform(0, total_weight)
        current = 0
        
        for i, (prompt, score) in enumerate(ranked):
            current += weights[i]
            if current >= pick:
                return (prompt, score)
        
        return ranked[-1]
    
    def stochastic_universal_sampling(self, scored_population: List[Tuple[str, float]], num_select: int) -> List[Tuple[str, float]]:
        """Stochastic Universal Sampling: More fair distribution with multiple selection points"""
        total_fitness = sum(score for _, score in scored_population)
        if total_fitness == 0:
            return random.sample(scored_population, min(num_select, len(scored_population)))
        
        # Adjust for negative fitness
        min_fitness = min(score for _, score in scored_population)
        if min_fitness < 0:
            adjusted_scores = [(prompt, score - min_fitness + 0.1) for prompt, score in scored_population]
        else:
            adjusted_scores = [(prompt, score + 0.1) for prompt, score in scored_population]
        
        total_adjusted = sum(score for _, score in adjusted_scores)
        distance = total_adjusted / num_select
        start = random.uniform(0, distance)
        
        pointers = [start + i * distance for i in range(num_select)]
        selected = []
        
        current = 0
        i = 0
        
        for prompt, score in adjusted_scores:
            current += score
            while i < len(pointers) and current >= pointers[i]:
                selected.append((prompt, score))
                i += 1
                if i >= num_select:
                    break
            if i >= num_select:
                break
        
        # Fill remaining slots if needed
        while len(selected) < num_select:
            selected.append(random.choice(scored_population))
        
        return selected[:num_select]
    
    def select_parents(self, scored_population: List[Tuple[str, float]]) -> Tuple[str, float]:
        """Select parents based on the chosen selection method"""
        if self.selection_method == "tournament":
            return self.tournament_selection(scored_population)
        elif self.selection_method == "roulette":
            return self.roulette_wheel_selection(scored_population)
        elif self.selection_method == "rank":
            return self.rank_selection(scored_population)
        elif self.selection_method == "sus":
            return self.stochastic_universal_sampling(scored_population, 1)[0]
        else:
            return self.tournament_selection(scored_population)  # Default fallback

class AutoPromptGA(GeneticAlgorithmBase):
    """AutoPrompt implementation using Genetic Algorithm with gradient-like word selection"""
    
    def __init__(self, model_pipeline, dataset: List[Dict], selection_method: str = "tournament"):
        super().__init__(model_pipeline, dataset, f"AutoPrompt-GA-{selection_method}", selection_method=selection_method)
        
        # Word bank for genetic operations (gradient-inspired vocabulary)
        self.word_bank = [
            "carefully", "step", "methodically", "precisely", "systematically",
            "logically", "thoroughly", "clearly", "detailed", "accurate",
            "solve", "calculate", "analyze", "think", "reason", "explain",
            "show", "work", "through", "problem", "solution", "answer",
            "check", "verify", "double-check", "confirm", "ensure"
        ]
    
    def mutate_prompt(self, prompt: str) -> str:
        """Gradient-inspired mutation: add/remove/replace words strategically"""
        words = prompt.split()
        
        mutation_type = random.choice(["add", "remove", "replace"])
        
        if mutation_type == "add" and len(words) < 20:
            # Add a beneficial word from word bank
            new_word = random.choice(self.word_bank)
            position = random.randint(0, len(words))
            words.insert(position, new_word)
            
        elif mutation_type == "remove" and len(words) > 3:
            # Remove a random word (but keep essential structure)
            words.pop(random.randint(0, len(words) - 1))
            
        elif mutation_type == "replace" and words:
            # Replace a word with one from word bank
            position = random.randint(0, len(words) - 1)
            words[position] = random.choice(self.word_bank)
        
        return " ".join(words)
    
    def crossover(self, parent1: str, parent2: str) -> str:
        """Smart crossover combining best parts of two prompts"""
        words1, words2 = parent1.split(), parent2.split()
        
        # Create hybrid by taking alternating segments
        min_len = min(len(words1), len(words2))
        cut_point = random.randint(1, max(1, min_len - 1))
        
        child_words = words1[:cut_point] + words2[cut_point:]
        return " ".join(child_words)
    
    def optimize(self, generations: int = 8) -> Tuple[str, float]:
        """AutoPrompt optimization using genetic algorithm"""
        
        # Initialize population
        population = [
            "Solve step by step and explain your reasoning clearly.",
            "Calculate carefully and show all your work methodically.",
            "Think through this problem systematically and precisely.",
            "Analyze the problem thoroughly and provide detailed solution.",
            "Work through this logically with clear explanations.",
            "Break down the problem and solve each step carefully.",
            "Think step by step and double-check your calculations.",
            "Solve methodically and verify your answer is accurate."
        ]
        
        for gen in range(generations):
            print(f"\n--- AutoPrompt Generation {gen+1}/{generations} ---")
            
            # Evaluate fitness for all individuals
            scored = [(prompt, self.evaluate_prompt(prompt)) for prompt in population]
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # Track metrics
            current_best = scored[0][1]
            self.fitness_history.append(current_best)
            self.diversity_history.append(self.calculate_diversity(population))
            
            print(f"Best score: {current_best:.3f} | Diversity: {self.diversity_history[-1]:.3f}")
            
            # Selection and reproduction
            new_population = []
            
            # Keep some survivors based on selection method
            survivors_count = max(2, self.population_size // 4)
            
            for _ in range(survivors_count):
                survivor = self.select_parents(scored)
                new_population.append(survivor[0])
            
            # Generate offspring through crossover and mutation
            while len(new_population) < self.population_size:
                parent1 = self.select_parents(scored)
                parent2 = self.select_parents(scored)
                
                child = self.crossover(parent1[0], parent2[0])
                
                # Apply mutation with probability
                if random.random() < 0.4:
                    child = self.mutate_prompt(child)
                
                new_population.append(child)
            
            population = new_population[:self.population_size]
        
        # Return best prompt found
        final_scored = [(prompt, self.evaluate_prompt(prompt)) for prompt in population]
        best_prompt, best_score = max(final_scored, key=lambda x: x[1])
        
        return best_prompt, best_score
