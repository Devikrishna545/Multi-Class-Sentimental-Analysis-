"""
Main entry point for training the sentiment analysis model
"""

from src.pipelines.training_pipeline import TrainingPipeline


def main():
    """Run the training pipeline"""
    print("Starting training pipeline...")
    pipeline = TrainingPipeline()
    results = pipeline.run()
    print(f"\nTraining completed! Final accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
