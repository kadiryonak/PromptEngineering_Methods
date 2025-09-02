
@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for prompt optimization methods"""
    
    accuracy: float = 0.0
    convergence_speed: float = 0.0  # generations to reach best score
    diversity_maintained: float = 0.0  # genetic diversity over generations
    computational_cost: float = 0.0  # time in seconds
    stability: float = 0.0  # consistency across runs
    final_score: float = 0.0
    
    def to_dict(self):
        return {
            'accuracy': self.accuracy,
            'convergence_speed': self.convergence_speed,
            'diversity_maintained': self.diversity_maintained,
            'computational_cost': self.computational_cost,
            'stability': self.stability,
            'final_score': self.final_score
        }
