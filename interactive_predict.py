"""
Interactive Sentiment Analysis CLI
Enter text to get real-time sentiment predictions
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.pipelines.inference_pipeline import InferencePipeline


class InteractiveSentimentAnalyzer:
    """Interactive CLI for sentiment analysis"""
    
    def __init__(self):
        """Initialize the inference pipeline"""
        print("\n" + "="*70)
        print(" "*15 + "INTERACTIVE SENTIMENT ANALYZER")
        print("="*70)
        print("\nInitializing model... Please wait...")
        
        try:
            self.pipeline = InferencePipeline()
            print("✓ Model loaded successfully!")
        except FileNotFoundError:
            print("\n❌ ERROR: Model not found!")
            print("Please train the model first by running: python train.py")
            sys.exit(1)
    
    def print_header(self):
        """Print welcome header"""
        print("\n" + "="*70)
        print("Welcome! Enter text to analyze its sentiment.")
        print("Commands:")
        print("  - Type 'quit' or 'exit' to close the application")
        print("  - Type 'help' for more information")
        print("  - Type 'examples' to see sample predictions")
        print("="*70 + "\n")
    
    def print_help(self):
        """Print help information"""
        print("\n" + "="*70)
        print(" "*25 + "HELP INFORMATION")
        print("="*70)
        print("\nThis tool analyzes the sentiment of text as:")
        print("  • POSITIVE - Expresses happiness, satisfaction, or approval")
        print("  • NEGATIVE - Expresses sadness, dissatisfaction, or disapproval")
        print("  • NEUTRAL  - Neither clearly positive nor negative")
        print("\nThe model provides:")
        print("  1. Predicted sentiment label")
        print("  2. Confidence score (0-100%)")
        print("  3. Probability distribution across all sentiments")
        print("  4. Text features (length, word count, symbols)")
        print("="*70 + "\n")
    
    def show_examples(self):
        """Show example predictions"""
        print("\n" + "="*70)
        print(" "*22 + "EXAMPLE PREDICTIONS")
        print("="*70)
        
        examples = [
            "This product is amazing! I love it!",
            "Terrible experience, would not recommend.",
            "The service was okay, nothing special.",
            "Best purchase I've ever made! Highly recommend!",
            "Disappointed with the quality."
        ]
        
        print("\nAnalyzing example sentences...\n")
        
        for i, text in enumerate(examples, 1):
            result = self.analyze_text(text, show_details=False)
            
            print(f"{i}. \"{text}\"")
            print(f"   → {result['predicted_sentiment'].upper()} "
                  f"({result['confidence']*100:.1f}% confidence)")
            print()
        
        print("="*70 + "\n")
    
    def analyze_text(self, text: str, show_details: bool = True):
        """
        Analyze sentiment of input text
        
        Args:
            text: Input text to analyze
            show_details: Whether to show detailed results
        
        Returns:
            Dictionary with prediction results
        """
        # Get prediction
        results = self.pipeline.predict_with_details(text)
        result = results[0]  # Get first result
        
        if show_details:
            # Display results
            print("\n" + "─"*70)
            print("ANALYSIS RESULTS")
            print("─"*70)
            print(f"\n📝 Original Text:")
            print(f"   \"{result['original_text']}\"")
            
            print(f"\n🔍 Cleaned Text:")
            print(f"   \"{result['cleaned_text']}\"")
            
            # Sentiment with emoji
            sentiment_emoji = {
                'positive': '😊',
                'negative': '😞',
                'neutral': '😐'
            }
            sentiment = result['predicted_sentiment'].lower()
            emoji = sentiment_emoji.get(sentiment, '🤔')
            
            print(f"\n{emoji} PREDICTED SENTIMENT: {result['predicted_sentiment'].upper()}")
            print(f"   Confidence: {result['confidence']*100:.2f}%")
            
            # Probability bar chart
            print(f"\n📊 Probability Distribution:")
            probs = result['probabilities']
            max_bar_width = 50
            
            for label in ['negative', 'neutral', 'positive']:
                prob = probs[label]
                bar_width = int(prob * max_bar_width)
                bar = '█' * bar_width + '░' * (max_bar_width - bar_width)
                print(f"   {label.capitalize():8s} [{bar}] {prob*100:5.2f}%")
            
            # Text features
            print(f"\n📈 Text Features:")
            features = result['features']
            print(f"   • Text Length:  {features['text_length']} characters")
            print(f"   • Word Count:   {features['word_count']} words")
            print(f"   • Symbol Count: {features['symbol_count']} symbols")
            
            print("─"*70 + "\n")
        
        return result
    
    def run(self):
        """Run the interactive loop"""
        self.print_header()
        
        while True:
            try:
                # Get user input
                user_input = input("Enter text to analyze (or command): ").strip()
                
                # Handle empty input
                if not user_input:
                    print("⚠️  Please enter some text to analyze.\n")
                    continue
                
                # Handle commands
                command = user_input.lower()
                
                if command in ['quit', 'exit', 'q']:
                    print("\n" + "="*70)
                    print("Thank you for using the Sentiment Analyzer!")
                    print("="*70 + "\n")
                    break
                
                elif command == 'help':
                    self.print_help()
                    continue
                
                elif command == 'examples':
                    self.show_examples()
                    continue
                
                elif command == 'clear':
                    # Clear screen (works on both Windows and Unix)
                    import os
                    os.system('cls' if os.name == 'nt' else 'clear')
                    self.print_header()
                    continue
                
                # Analyze the text
                self.analyze_text(user_input)
                
            except KeyboardInterrupt:
                print("\n\n" + "="*70)
                print("Session interrupted by user.")
                print("="*70 + "\n")
                break
            
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try again or type 'help' for assistance.\n")


def main():
    """Main entry point"""
    analyzer = InteractiveSentimentAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    main()
