
# -----------------------------
# Main Execution
# -----------------------------
def main():
    """Main execution function"""
    
    print("Loading model...")
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    # Extended evaluation dataset
    dataset = [
        {"q": "What is 12 + 7?", "a": "19"},
        {"q": "If a book costs 5 dollars, how much for 3 books?", "a": "15"},
        {"q": "A car travels 80 km/h for 2 hours. How far does it go?", "a": "160"},
        {"q": "What is 25% of 80?", "a": "20"},
        {"q": "If 3 apples cost 6 dollars, how much does 1 apple cost?", "a": "2"},
        {"q": "What is 45 - 28?", "a": "17"},
        {"q": "A rectangle has length 8 and width 5. What is its area?", "a": "40"},
        {"q": "If you save 10 dollars per week, how much in 6 weeks?", "a": "60"}
    ]
    
    print(f"Dataset loaded with {len(dataset)} questions")
    
    # Initialize benchmark
    benchmark = PromptOptimizationBenchmark(pipe, dataset)
    
    # Run comprehensive comparison
    benchmark.run_benchmark(runs_per_method=2, generations=5)  # Reduced for faster execution
    
    # Display results
    best_method = benchmark.print_comparison_table()
    
    print(f"\n🏆 WINNER: {best_method[0]} with overall score: {best_method[1]:.4f}")
    
    return benchmark.results

# -----------------------------
# Execution Guard
# -----------------------------
if __name__ == "__main__":
    results = main()
    
    # Optional: Save results to JSON for further analysis
    # with open('prompt_optimization_results.json', 'w') as f:
    #     json_results = {method: metrics.to_dict() for method, metrics in results.items()}
    #     json.dump(json_results, f, indent=2)