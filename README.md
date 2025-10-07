
# 🎬 IMDb Sentiment Analysis

A Natural Language Processing (NLP) project that classifies IMDb movie reviews as **positive** or **negative** using two supervised learning models: **Logistic Regression** and **Multinomial Naive Bayes**.  
This project compares both models and provides an interface for quick predictions using pre-trained models.

## 🚀 Overview

The dataset used for this project is the IMDb reviews dataset, which contains 50,000 labeled reviews (25k for training, 25k for testing).  
Each review is preprocessed (lowercasing, HTML removal, punctuation cleaning, and stopword filtering) before being vectorized and classified.

### Models Used
- **Logistic Regression + TF-IDF Vectorizer**
- **Multinomial Naive Bayes + CountVectorizer**

The Logistic Regression model performed better overall, achieving higher accuracy and balanced precision/recall scores.

## 📊 Model Performance

| Metric | Logistic Regression | Multinomial NB |
|--------|---------------------|----------------|
| Accuracy | 0.8834 | 0.8241 |
| Precision | 0.884 | 0.865 |
| Recall | 0.882 | 0.767 |
| F1-score | 0.883 | 0.813 |

## 💡 How It Works

1. Load the pre-trained vectorizers and models (`.pkl` files).  
2. Input a movie review (string).  
3. The model predicts whether the sentiment is **Positive (1)** or **Negative (0)**.

## 🧠 Technologies Used

- Python 3.13.5
- Pandas  
- NumPy  
- Scikit-learn  
- NLTK  
- Joblib / Pickle  
- Matplotlib & Seaborn (for visualization)  

## ⚙️ Setup Instructions

```bash
git clone https://github.com/JuanDavidQuesada/IMDb-Sentiment-Analysis.git
cd IMDb-Sentiment-Analysis
pip install -r requirements.txt
python main.py
```

## 📄 Files Included

- `main.py` – Code for loading models and predicting new reviews.  
- `requirements.txt` – Dependencies for running the notebook or script.  
- `LogisticRegression_model.pkl` / `MultinomialNB_model.pkl` – Pre-trained models.  
- `*_vectorizer.pkl` – Corresponding vectorizers for text transformation.

## 👨‍💻 Author

**Juan David Quesada Estrada**  
[LinkedIn](https://www.linkedin.com/in/juan-david-quesada-estrada-1011521b4/)  
[GitHub](https://github.com/JuanDavidQuesada)


