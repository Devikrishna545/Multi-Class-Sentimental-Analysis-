"""
Main entry point for making predictions
"""

from src.pipelines.inference_pipeline import InferencePipeline


def main():
    """Run the inference pipeline with sample texts"""
    print("Starting inference pipeline...")
    
    pipeline = InferencePipeline()
    
    # Sample texts for demonstration
    sample_texts = [
        "The product was excellent and exceeded my expectations!",
        "This movie is terrible, I hate it.",
        "The service was okay, nothing special.",
        "I absolutely love this! Best purchase ever!",
        "Disappointed with the quality, would not recommend."
    ]
    
    print("\nMaking predictions on sample texts...")
    results = pipeline.predict_and_display(sample_texts)
    
    return results


if __name__ == "__main__":
    main()
