"""
Main entry point for model evaluation
"""

from src.pipelines.evaluation_pipeline import EvaluationPipeline


def main():
    """Run the evaluation pipeline"""
    print("Starting evaluation pipeline...")
    
    pipeline = EvaluationPipeline()
    results = pipeline.run_comprehensive_evaluation()
    
    print(f"\nEvaluation completed! Accuracy: {results['accuracy']:.4f}")
    
    return results


if __name__ == "__main__":
    main()
