# 📰 News Classifier

A sophisticated web application that uses machine learning to classify news articles into categories and perform natural language processing tasks.


## 🌟 Features

- **News Classification**: Categorize news articles into business, tech, sports, health, politics, or entertainment
- **Text Analysis**: Powerful NLP tools for text processing
  - Tokenization
  - Lemmatization
  - Named Entity Recognition
  - Part-of-Speech Tagging
- **Text Visualization**:
  - Interactive Word Clouds
  - Word Frequency Analysis
  - Text Statistics
- **Theming**: Toggle between light and dark modes for comfortable viewing
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Streamlit
- spaCy (with English language model)
- Scikit-learn
- Pandas
- Matplotlib
- WordCloud
- Pillow (PIL)

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/news-classifier.git
   cd news-classifier
   ```

2. Install required packages
   ```bash
   pip install streamlit joblib os pandas wordcloud matplotlib PIL
   ```

3. Download the spaCy English model
   ```bash
   python -m spacy download en
   ```

4. Make sure you have the model files in the `models/` directory:
   - `final_news_cv_vectorizer.pkl`
   - `newsclassifier_Logit_model.pkl`
   - `newsclassifier_RFOREST_model.pkl`
   - `newsclassifier_NB_model.pkl`
   - `newsclassifier_CART_model.pkl`

### Running the App

```bash
streamlit run app.py
```

Navigate to the URL provided in the terminal (typically http://localhost:8501).

## 🔍 How It Works

### News Classification

1. Enter or paste a news article in the text area
2. Select a machine learning algorithm:
   - Logistic Regression
   - Random Forest
   - Naive Bayes
   - Decision Tree
3. Click "Classify News" to categorize the article

### Text Analysis

1. Enter or paste text in the text area
2. Select an NLP task:
   - Tokenization: Break text into individual words/tokens
   - Lemmatization: Reduce words to their base form
   - Named Entities: Identify and classify named entities (people, organizations, locations, etc.)
   - POS Tags: Identify grammatical parts of speech
3. View results in formatted output or tables

### Visualization

1. Enter or paste text in the text area
2. Select visualization types:
   - Word Cloud: Visual representation of word frequency
   - Word Frequency: Bar chart of most common words
   - Character Count: Text statistics including characters, words, and sentences
3. Generate and view visualizations

## 🧠 Machine Learning Models

The app includes four different classification models:

1. **Logistic Regression**: Fast and efficient for text classification
2. **Random Forest**: Ensemble method with high accuracy
3. **Naive Bayes**: Probabilistic classifier well-suited for text
4. **Decision Tree**: Simple, interpretable model

Models were trained on a dataset of labeled news articles across six categories.

## 🎨 Customization

### Theming

Toggle between light and dark modes using the button in the sidebar. The app remembers your preference during your session.

### Styling

The app uses custom CSS for styling. You can modify the styles in the `local_css()` function.

## 📁 Project Structure

```
news-classifier/
├── app.py              # Main application file
├── models/             # Trained machine learning models
│   ├── final_news_cv_vectorizer.pkl
│   ├── newsclassifier_Logit_model.pkl
│   ├── newsclassifier_RFOREST_model.pkl
│   ├── newsclassifier_NB_model.pkl
│   └── newsclassifier_CART_model.pkl
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 🔧 Technical Details

- **Frontend**: Streamlit for UI components and interactivity
- **Backend**: Python for processing and machine learning
- **NLP**: spaCy for natural language processing tasks
- **ML Models**: scikit-learn models for classification
- **Data Visualization**: Matplotlib and WordCloud for visualizations

## 📊 Training Your Own Models

To train your own models:

1. Prepare a labeled dataset of news articles
2. Use scikit-learn to train classification models
3. Save models using joblib
4. Replace the existing model files in the `models/` directory

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web app framework
- [spaCy](https://spacy.io/) for NLP capabilities
- [scikit-learn](https://scikit-learn.org/) for machine learning models
