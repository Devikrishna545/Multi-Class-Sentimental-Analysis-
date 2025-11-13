"""
Test pipelines
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def test_inference_pipeline():
    """Test inference pipeline"""
    from src.pipelines.inference_pipeline import InferencePipeline
    
    try:
        pipeline = InferencePipeline()
        
        # Test single prediction
        text = "This product is amazing!"
        predictions, probabilities = pipeline.predict(text)
        
        assert len(predictions) == 1, "Should have one prediction"
        assert probabilities is not None, "Should have probabilities"
        
        print("✓ Inference pipeline test passed")
    except FileNotFoundError:
        print("⚠ Inference pipeline test skipped (model not trained yet)")


if __name__ == "__main__":
    print("Running pipeline tests...")
    test_inference_pipeline()
    print("\nPipeline tests completed!")
