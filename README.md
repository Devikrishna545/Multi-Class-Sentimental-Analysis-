# Sentiment Analysis Project

A comprehensive machine learning project for multi-class sentiment analysis using text data from Hugging Face datasets with multiple algorithm comparison and feature engineering.

## 📋 Project Overview

This project implements a sentiment analysis classifier that predicts sentiment (positive, negative, neutral) from text data. The project explores multiple machine learning approaches and achieves **66.15% accuracy** with Logistic Regression as the best-performing model.

## 🎯 Key Features

- **Multi-class sentiment classification** (positive, negative, neutral)
- **Advanced text preprocessing** with stopword removal and tokenization
- **Comprehensive feature engineering**:
  - Bag of Words (BOW) representation
  - Text length analysis
  - Word count analysis  
  - Symbol/punctuation count analysis
- **Complete dataset utilization** (proper train/test split from Hugging Face)
- **Multiple algorithm comparison**:
  - Logistic Regression (66.15% accuracy - **BEST**)
  - Random Forest
  - Support Vector Machine (SVM)
  - Multi-Layer Perceptron (MLP)
- **Data visualization and analysis**
- **One-hot encoding for categorical features**

## 📊 Dataset

The project uses the [multiclass-sentiment-analysis-dataset](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset) from Hugging Face:

- **Training set**: 31,232 samples
- **Test set**: 5,205 samples
- **Features**: text, sentiment labels, and engineered numeric features
- **Classes**: negative, neutral, positive (balanced distribution)

## 🛠️ Technology Stack

- **Python 3.x**
- **Core Libraries**:
  - `pandas` - Data manipulation and analysis
  - `scikit-learn` - Machine learning algorithms and metrics
  - `nltk` - Natural language processing and stopwords
  - `matplotlib` & `seaborn` - Data visualization
  - `huggingface_hub` - Dataset access
  - `scipy` - Sparse matrix operations
  - `numpy` - Numerical computing
  - `python-dotenv` - Environment variable management

## 🚀 Getting Started

### Prerequisites

1. **Python 3.7 or higher**
2. **Hugging Face account and token**
3. **Jupyter Notebook or VS Code with Python extension**

### Installation

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd Juproject
   ```

2. Install required packages:
   ```bash
   pip install pandas scikit-learn nltk matplotlib seaborn huggingface_hub python-dotenv scipy numpy
   ```

3. Download NLTK data:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

### Setup

1. Create a `.env` file in the project root:
   ```
   secret_token_hugface=your_huggingface_token_here
   ```

2. Replace `your_huggingface_token_here` with your actual Hugging Face token.

## 📝 Usage

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook Sentimental_Analysis_.ipynb
   ```
   
   Or open in VS Code:
   ```bash
   code Sentimental_Analysis_.ipynb
   ```

2. Run all cells in sequence to:
   - Load and preprocess the data
   - Engineer features (text length, word count, symbol count)
   - Visualize data distributions
   - Train multiple models and compare performance
   - Evaluate on the test set

## 🏗️ Project Structure

```
Juproject/
├── Sentimental_Analysis_.ipynb    # Main notebook with complete analysis
├── .gitignore                     # Git ignore file
├── README.md                      # Project documentation
└── .env                          # Environment variables (create this)
```

## 🧠 Implementation Details

### Data Preprocessing Pipeline
1. **Data Cleaning**: Remove duplicates and null values
2. **Text Preprocessing**: 
   - Remove stopwords using NLTK
   - Tokenization with WordPunctTokenizer
   - Remove punctuation and special characters
3. **Feature Engineering**:
   - One-hot encoding for sentiment labels
   - Text length calculation
   - Word count per text
   - Symbol/punctuation count
4. **Vectorization**: Bag of Words using CountVectorizer

### Model Architecture & Results

| Algorithm | Accuracy | Notes |
|-----------|----------|-------|
| **Logistic Regression** | **66.15%** | **Best performing model** |
| Random Forest | ~60-65% | Good performance, robust |
| Support Vector Machine | ~55-60% | Decent performance |
| Multi-Layer Perceptron | ~54.68% | Initial baseline model |

### Feature Engineering Details
- **Text Features**: ~28,912 BOW features from preprocessed text
- **Numeric Features**: 3 engineered features (text_length, word_count, symbol_count)
- **Total Features**: ~28,915 combined features
- **Target Encoding**: Label encoding (0=negative, 1=neutral, 2=positive)

## 📈 Results & Performance

### Best Model Performance
- **Algorithm**: Logistic Regression
- **Accuracy**: 66.15%
- **Training Samples**: 31,232 (complete training set)
- **Test Samples**: 5,205 (separate test set)

### Key Insights
1. **Logistic Regression outperformed complex models** - sometimes simpler is better
2. **Feature engineering significantly improved performance** - text statistics matter
3. **Proper preprocessing is crucial** - stopword removal and tokenization helped
4. **Complete dataset usage** - no data waste with proper train/test splits

## 🔧 Data Visualization

The notebook includes comprehensive visualizations:
- **Text length distribution** by sentiment
- **Word count analysis** across different sentiments  
- **Symbol count patterns** in positive/negative/neutral texts
- **Dataset statistics and distributions**

## 🎓 Learning Outcomes

This project demonstrates:
- **Multi-class classification techniques**
- **Feature engineering for text data**
- **Model comparison and selection**
- **Proper data preprocessing pipelines**
- **Real-world dataset handling from Hugging Face**
- **Performance optimization through algorithm selection**

## 🚀 Future Improvements

Potential enhancements to consider:
- **Advanced NLP techniques**: TF-IDF, n-grams, word embeddings
- **Deep learning models**: BERT, RoBERTa, or custom neural networks
- **Ensemble methods**: Combining top-performing models
- **Hyperparameter tuning**: Grid search or random search optimization
- **Cross-validation**: More robust model evaluation
- **Additional features**: Sentiment lexicons, part-of-speech tags

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Hugging Face** for providing the comprehensive sentiment analysis dataset
- **NLTK team** for excellent text processing tools
- **Scikit-learn** for robust machine learning algorithms
- **Pandas & NumPy** for efficient data manipulation
- **Matplotlib & Seaborn** for beautiful data visualizations

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

## 📊 Quick Results Summary

```
📈 PERFORMANCE COMPARISON:
┌─────────────────────┬──────────┬─────────────────┐
│ Algorithm           │ Accuracy │ Status          │
├─────────────────────┼──────────┼─────────────────┤
│ Logistic Regression │  66.15%  │ 🏆 BEST MODEL  │
│ Random Forest       │  ~60-65% │ ✅ Good         │
│ SVM                 │  ~55-60% │ ✅ Decent       │
│ MLP Neural Network  │  54.68%  │ ✅ Baseline     │
└─────────────────────┴──────────┴─────────────────┘

🎯 Dataset: 31,232 training + 5,205 test samples
🔧 Features: 28,915 (BOW + engineered numeric features)
📝 Classes: negative, neutral, positive
```

---

**Note**: Make sure to keep your Hugging Face token secure and never commit it to version control. The `.env` file is included in `.gitignore` for your security.