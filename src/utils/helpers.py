"""
Utility helper functions
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def save_json(data: Dict[str, Any], filepath: str):
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Data saved to {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary from JSON
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data


def create_timestamp() -> str:
    """
    Create timestamp string
    
    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_results(results: Dict[str, Any]) -> str:
    """
    Format evaluation results for display
    
    Args:
        results: Results dictionary
        
    Returns:
        Formatted string
    """
    output = []
    output.append("="*50)
    output.append("MODEL EVALUATION RESULTS")
    output.append("="*50)
    output.append(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    output.append("")
    
    # Add classification report details if available
    if 'classification_report' in results:
        output.append("Per-Class Metrics:")
        for label, metrics in results['classification_report'].items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                output.append(f"\n{label.upper()}:")
                output.append(f"  Precision: {metrics['precision']:.4f}")
                output.append(f"  Recall:    {metrics['recall']:.4f}")
                output.append(f"  F1-Score:  {metrics['f1-score']:.4f}")
    
    return "\n".join(output)
