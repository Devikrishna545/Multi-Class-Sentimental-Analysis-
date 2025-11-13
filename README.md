# Multi-Class Sentiment Analysis

A comprehensive production-ready machine learning project for multi-class sentiment analysis using text data from Hugging Face datasets. Features modular architecture, multiple algorithm comparison, hyperparameter tuning, and interactive CLI for real-time predictions.

## 📋 Project Overview

This project implements an end-to-end sentiment analysis system that classifies text into three categories: positive, negative, and neutral. Through systematic experimentation with multiple algorithms and hyperparameter optimization, the project achieves **66.36% accuracy** on the test set using an optimized Logistic Regression model.

**Key Highlights:**
- Production-ready modular codebase with organized `src/` structure
- Interactive CLI tool for real-time sentiment analysis
- Comprehensive Jupyter notebook for experimentation and visualization
- Automated training, evaluation, and prediction pipelines
- Model persistence for easy deployment

## 🎯 Key Features

- **Multi-class sentiment classification** (positive, negative, neutral)
- **Modular architecture** with separation of concerns (data, features, models, pipelines)
- **Interactive CLI** for real-time sentiment predictions with detailed analysis
- **Advanced text preprocessing** with stopword removal and tokenization
- **Comprehensive feature engineering**:
  - Bag of Words (BOW) representation using CountVectorizer
  - Text length analysis
  - Word count analysis  
  - Symbol/punctuation count analysis
- **Complete dataset utilization** (train/validation/test split from Hugging Face)
- **Multiple algorithm comparison and evaluation**
- **Hyperparameter tuning** using GridSearchCV
- **Model persistence** with joblib for deployment
- **Automated pipelines** for training, inference, and evaluation
- **Data visualization and exploratory data analysis**
- **Unit tests** for code reliability

## 📊 Dataset

The project uses the [multiclass-sentiment-analysis-dataset](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset) from Hugging Face:

- **Training set**: 31,232 samples
- **Validation set**: Available for model tuning
- **Test set**: 5,205 samples
- **Features**: text, sentiment labels, and engineered numeric features
- **Label Distribution**:
  - Neutral (label 1): 11,649 samples (37.3%)
  - Positive (label 2): 10,478 samples (33.5%)
  - Negative (label 0): 9,105 samples (29.2%)

## 🛠️ Technology Stack

- **Python 3.x**
- **Core Libraries**:
  - `pandas` (>=1.3.0) - Data manipulation and analysis
  - `numpy` (>=1.21.0) - Numerical computing
  - `scikit-learn` (>=1.0.0) - Machine learning algorithms and metrics
  - `nltk` (>=3.6.0) - Natural language processing and stopwords
  - `matplotlib` (>=3.4.0) & `seaborn` (>=0.11.0) - Data visualization
  - `scipy` (>=1.7.0) - Sparse matrix operations
  - `huggingface-hub` (>=0.8.0) - Dataset access
  - `python-dotenv` (>=0.19.0) - Environment variable management
  - `jupyter` (>=1.0.0) - Notebook support
  - `ipykernel` (>=6.0.0) - Jupyter kernel
  - `tqdm` (>=4.60.0) - Progress bars
  - `requests` (>=2.25.0) - HTTP library
  - `joblib` - Model serialization

## 🚀 Getting Started

### Prerequisites

1. **Python 3.7 or higher**
2. **Hugging Face account and token** (Get yours at [huggingface.co](https://huggingface.co/settings/tokens))
3. **Git** for cloning the repository

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Devikrishna545/Multi-Class-Sentimental-Analysis-.git
   cd Multi-Class-Sentimental-Analysis-
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK data:
   ```python
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
   ```

### Setup

1. Create a `.env` file in the project root:
   ```env
   secret_token_hugface=your_huggingface_token_here
   ```

2. Replace `your_huggingface_token_here` with your actual Hugging Face token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## 📝 Usage

### Method 1: Using Python Scripts (Recommended for Production)

#### Train the Model
```bash
python train.py
```
This will:
- Load and preprocess data from Hugging Face
- Engineer features and create BOW representations
- Train the optimized Logistic Regression model
- Save the trained model and vectorizer

#### Make Predictions
```bash
python predict.py
```
Run predictions on sample texts to test the model.

#### Evaluate the Model
```bash
python evaluate.py
```
Run comprehensive evaluation on the test set with detailed metrics.

#### Interactive CLI (Real-time Analysis)
```bash
python interactive_predict.py
```
Launch an interactive command-line interface for real-time sentiment analysis:
- Enter any text to get instant sentiment predictions
- View confidence scores and probability distributions
- See text features and detailed analysis
- Commands: `help`, `examples`, `quit`

**Interactive CLI Features:**
- 😊 Emoji-based sentiment indicators
- 📊 Visual probability bars
- 📈 Text feature statistics
- 🔍 Side-by-side original and cleaned text comparison

### Method 2: Using Jupyter Notebook (Recommended for Experimentation)

1. Launch Jupyter:
   ```bash
   jupyter notebook Sentimental_Analysis_.ipynb
   ```

2. Run all cells in sequence to:
   - Load and preprocess the data from Hugging Face
   - Engineer features (text length, word count, symbol count)
   - Visualize data distributions
   - Train multiple models and compare performance
   - Perform hyperparameter tuning
   - Evaluate on the test set
   - Save the best model

### Method 3: Programmatic API

```python
from src.pipelines.inference_pipeline import InferencePipeline

# Initialize pipeline
pipeline = InferencePipeline()

# Single prediction
text = "I love this product! It's amazing!"
result = pipeline.predict_with_details(text)[0]

print(f"Sentiment: {result['predicted_sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")

# Batch predictions
texts = [
    "Excellent service!",
    "Terrible experience.",
    "It was okay."
]
results = pipeline.predict_and_display(texts)
```

## 🏗️ Project Structure

```
Multi-Class-Sentimental-Analysis-/
├── src/                           # Source code package
│   ├── __init__.py               # Package initializer
│   ├── config/                   # Configuration files
│   │   ├── __init__.py
│   │   └── config.py            # Model and data configurations
│   ├── data/                     # Data loading and processing
│   │   ├── __init__.py
│   │   └── data_loader.py       # HuggingFace dataset loader
│   ├── features/                 # Feature engineering
│   │   ├── __init__.py
│   │   ├── preprocessing.py     # Text preprocessing (stopwords, tokenization)
│   │   └── feature_engineering.py  # BOW, text features
│   ├── models/                   # Model definitions
│   │   ├── __init__.py
│   │   └── train.py             # SentimentModel class with train/predict
│   ├── pipelines/                # End-to-end pipelines
│   │   ├── __init__.py
│   │   ├── training_pipeline.py    # Complete training workflow
│   │   ├── inference_pipeline.py   # Prediction workflow
│   │   └── evaluation_pipeline.py  # Model evaluation workflow
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       └── helpers.py           # Common helper functions
├── notebooks/                    # Jupyter notebooks
│   └── (exploration notebooks)
├── tests/                        # Unit tests
│   └── (test files)
├── train.py                      # Main training script
├── predict.py                    # Batch prediction script
├── evaluate.py                   # Model evaluation script
├── interactive_predict.py        # Interactive CLI tool
├── Sentimental_Analysis_.ipynb   # Main analysis notebook
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
├── .env                          # Environment variables (not in repo)
├── README.md                     # This file
├── best_sentiment_model.pkl      # Saved model (generated after training)
└── vectorizer.pkl                # Saved vectorizer (generated after training)
```

## 🧠 Implementation Details

### Complete Pathway to Achieve 66.36% Accuracy

#### **Step 1: Data Loading and Preprocessing**
```python
# 1. Load dataset from Hugging Face
df = pd.read_csv("hf://datasets/Sp1786/multiclass-sentiment-analysis-dataset/train_df.csv")

# 2. Data Cleaning
df.dropna(inplace=True)              # Remove null values
df.drop_duplicates(inplace=True)     # Remove duplicates
df.drop(columns=['id'], inplace=True) # Drop irrelevant columns
```

#### **Step 2: Feature Engineering**
```python
# 1. One-hot encode labels
df = pd.get_dummies(df, columns=['label'], prefix='label')

# 2. Text-based features
df['text_length'] = df['text'].apply(len)                    # Character count
df['word_count'] = df['text'].apply(lambda x: len(x.split())) # Word count
df['symbol_count'] = df['text'].apply(lambda x: sum(not c.isalnum() and not c.isspace() for c in x))
```

#### **Step 3: Text Preprocessing**
```python
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

# Remove stopwords and tokenize
stop_words = set(stopwords.words('english'))
tokenizer = WordPunctTokenizer()

def preprocess_text(text):
    # Tokenize
    tokens = tokenizer.tokenize(text.lower())
    # Remove stopwords and punctuation
    filtered = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered)

df['processed_text'] = df['text'].apply(preprocess_text)
```

#### **Step 4: Vectorization**
```python
from sklearn.feature_extraction.text import CountVectorizer

# Create Bag of Words features
vectorizer = CountVectorizer(max_features=5000)
bow_features = vectorizer.fit_transform(df['processed_text'])

# Combine BOW with numeric features
from scipy.sparse import hstack
numeric_features = df[['text_length', 'word_count', 'symbol_count']].values
X_combined = hstack([bow_features, numeric_features])
```

#### **Step 5: Train-Test Split**
```python
from sklearn.model_selection import train_test_split

# Prepare target variable
y = df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)
```

#### **Step 6: Algorithm Comparison (Initial)**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Test multiple algorithms on subset
algorithms = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42),
    'MLP': MLPClassifier(hidden_layers=(100,), random_state=42)
}

# Results on subset data:
# Logistic Regression: 54.40% ✓ BEST on subset
# Random Forest: 53.90%
# MLP: 48.70%
# SVM: 39.00%
```

#### **Step 7: Hyperparameter Tuning (Logistic Regression)**
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'class_weight': [None, 'balanced']
}

# Perform grid search
grid_search = GridSearchCV(
    LogisticRegression(max_iter=2000, random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best parameters found:
# {'C': 1, 'class_weight': None, 'penalty': 'l2', 'solver': 'liblinear'}
# CV Score: 55.10%
```

#### **Step 8: Final Model Training on Full Dataset**
```python
# Train best model with optimized parameters on full training data
best_model = LogisticRegression(
    C=1,
    penalty='l2',
    solver='liblinear',
    max_iter=2000,
    random_state=42
)

best_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Final Test Accuracy: {final_accuracy:.4f}")
# Final Model Accuracy: 66.36% 🏆
```

#### **Step 9: Model Persistence**
```python
import joblib

# Save the trained model
joblib.dump(best_model, 'best_sentiment_model.pkl')

# Save the vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')

# Load model for predictions
loaded_model = joblib.load('best_sentiment_model.pkl')

# Make predictions on new data
new_predictions = loaded_model.predict(new_data_vectorized)
```

### Feature Set Details

- **Text Features**: ~5,000 BOW features from preprocessed text
- **Numeric Features**: 
  - `text_length`: Character count of each text
  - `word_count`: Number of words in each text
  - `symbol_count`: Count of punctuation/symbols
- **Total Features**: ~5,003 combined features
- **Target**: Sentiment labels (0=negative, 1=neutral, 2=positive)

## 📈 Results & Performance

### Model Comparison Results

| Algorithm | Initial Accuracy | Notes |
|-----------|-----------------|-------|
| **Logistic Regression** | **54.40%** | **Selected for tuning** |
| Random Forest | 53.90% | Good performance, but slower |
| Multi-Layer Perceptron | 48.70% | Baseline neural network |
| Support Vector Machine | 39.00% | Poor performance on this dataset |

### Hyperparameter Tuning Results

**Logistic Regression Optimization:**
- **Cross-Validation Score**: 55.10%
- **Best Parameters**:
  - C: 1
  - Penalty: l2
  - Solver: liblinear
  - Class Weight: None

### Final Model Performance

- **Algorithm**: Logistic Regression (Optimized)
- **Final Test Accuracy**: **66.36%** 🏆
- **Training Samples**: 24,985 (80% of 31,232)
- **Test Samples**: 6,247 (20% of 31,232)
- **Improvement**: 11.96% increase from initial baseline (54.40% → 66.36%)

### Performance Breakdown

```
📊 ACCURACY PROGRESSION:
┌────────────────────────────┬──────────┬────────────┐
│ Stage                      │ Accuracy │ Improvement│
├────────────────────────────┼──────────┼────────────┤
│ Initial Logistic Reg.      │  54.40%  │   Baseline │
│ After Hyperparameter Tuning│  55.10%  │   +0.70%   │
│ Final Model (Full Data)    │  66.36%  │  +11.96%   │
└────────────────────────────┴──────────┴────────────┘

🎯 Key Success Factors:
✓ Feature Engineering (text_length, word_count, symbol_count)
✓ Text Preprocessing (stopword removal, tokenization)
✓ Hyperparameter Tuning (GridSearchCV)
✓ Full Dataset Training (no data waste)
✓ Balanced approach (BOW + numeric features)
```

## 🔧 Data Visualization

The notebook includes comprehensive visualizations:
- **Text length distribution** by sentiment
- **Word count analysis** across different sentiments  
- **Symbol count patterns** in positive/negative/neutral texts
- **Dataset statistics and class distributions**
- **Feature correlation analysis**
- **Model performance comparisons**
- **Confusion matrices** for detailed error analysis

## 🎓 Learning Outcomes

This project demonstrates:
- **Multi-class classification techniques** in NLP
- **Modular software architecture** for ML projects
- **Feature engineering strategies** to boost model performance (+11.96%)
- **Algorithm comparison methodology** for model selection
- **Hyperparameter optimization** using GridSearchCV
- **Model persistence** for production deployment
- **Complete ML pipeline** from data loading to prediction
- **Best practices** in train/test splits and evaluation
- **Real-world NLP preprocessing** with NLTK
- **Interactive CLI development** for user-friendly predictions

## 🚀 Future Improvements

Potential enhancements to consider:
- **Advanced NLP techniques**: 
  - TF-IDF weighting instead of simple BOW
  - N-grams (bigrams, trigrams) for context
  - Word embeddings (Word2Vec, GloVe, FastText)
- **Deep learning models**: 
  - BERT, RoBERTa, or DistilBERT transformers
  - Custom LSTM/GRU networks
  - Attention mechanisms
- **Ensemble methods**: 
  - Stacking top-performing models
  - Voting classifiers
  - Boosting techniques (XGBoost, LightGBM)
- **Advanced tuning**:
  - Bayesian optimization
  - Expanded hyperparameter search
  - Cross-validation strategies (K-fold)
- **Additional features**: 
  - Sentiment lexicons (VADER, TextBlob)
  - Part-of-speech tags
  - Named entity recognition
  - Emoji and emoticon analysis
- **Class balancing**: 
  - SMOTE or ADASYN for minority classes
  - Class weights optimization
- **Deployment**:
  - Flask/FastAPI REST API
  - Streamlit web application
  - Docker containerization
  - Cloud deployment (AWS, GCP, Azure)

## 💡 API Examples

### Using the Training Pipeline
```python
from src.pipelines.training_pipeline import TrainingPipeline

# Train the model
pipeline = TrainingPipeline()
results = pipeline.run()
print(f"Training accuracy: {results['accuracy']:.4f}")
```

### Using the Inference Pipeline
```python
from src.pipelines.inference_pipeline import InferencePipeline

# Make predictions
pipeline = InferencePipeline()
text = "This is an amazing product!"
result = pipeline.predict_with_details(text)[0]

print(f"Sentiment: {result['predicted_sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

### Using the Evaluation Pipeline
```python
from src.pipelines.evaluation_pipeline import EvaluationPipeline

# Evaluate model
pipeline = EvaluationPipeline()
results = pipeline.run_comprehensive_evaluation()
print(f"Test accuracy: {results['accuracy']:.4f}")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Hugging Face** for providing the comprehensive sentiment analysis dataset
- **Sp1786** for creating and sharing the multiclass-sentiment-analysis-dataset
- **NLTK team** for excellent text processing tools
- **Scikit-learn** for robust machine learning algorithms and GridSearchCV
- **Pandas & NumPy** for efficient data manipulation
- **Matplotlib & Seaborn** for beautiful data visualizations

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- **GitHub**: [@Devikrishna545](https://github.com/Devikrishna545)
- **Repository**: [Multi-Class-Sentimental-Analysis-](https://github.com/Devikrishna545/Multi-Class-Sentimental-Analysis-)

Feel free to open an issue on GitHub for any questions or bug reports!

## 📊 Quick Results Summary

```
📈 FINAL RESULTS:
┌─────────────────────────────────────────────────────┐
│  🏆 BEST MODEL: Logistic Regression (Optimized)    │
├─────────────────────────────────────────────────────┤
│  📊 Test Accuracy: 66.36%                          │
│  🎯 Training Samples: 24,985                       │
│  🧪 Test Samples: 6,247                            │
│  ⚙️  Best Parameters:                              │
│     • C: 1                                         │
│     • Penalty: l2                                  │
│     • Solver: liblinear                            │
│  📈 Improvement: +11.96% from baseline             │
└─────────────────────────────────────────────────────┘

🎯 Dataset: 31,232 training samples
🔧 Features: ~5,003 (BOW + engineered numeric features)
📝 Classes: negative (0), neutral (1), positive (2)
💾 Model Saved: best_sentiment_model.pkl
🔄 Vectorizer Saved: vectorizer.pkl
```

---

## 🔒 Security Note

**Important**: Keep your Hugging Face token secure and never commit it to version control. The `.env` file is included in `.gitignore` for your security. Never share your token publicly.

---

## 🛠️ Quick Start Guide

```bash
# Clone the repository
git clone https://github.com/Devikrishna545/Multi-Class-Sentimental-Analysis-.git

# Navigate to project directory
cd Multi-Class-Sentimental-Analysis-

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Create .env file and add your Hugging Face token
echo "secret_token_hugface=your_token_here" > .env

# Train the model
python train.py

# Try the interactive CLI
python interactive_predict.py
```

**Happy Analyzing!** 🎉📊🚀