
import pickle
import os

# Load models and vectorizers
def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

# File paths
base_dir = os.path.dirname(__file__)
logreg_model_path = os.path.join(base_dir, 'LogisticRegression_model.pkl')
logreg_vec_path = os.path.join(base_dir, 'LogisticRegression_vectorizer.pkl')
nb_model_path = os.path.join(base_dir, 'MultinomialNB_model.pkl')
nb_vec_path = os.path.join(base_dir, 'MultinomialNB_vectorizer.pkl')

logreg_model = load_pickle(logreg_model_path)
logreg_vectorizer = load_pickle(logreg_vec_path)
nb_model = load_pickle(nb_model_path)
nb_vectorizer = load_pickle(nb_vec_path)

# Predict sentiment using both models
def predict_sentiment(review):
    review_vectorized_lr = logreg_vectorizer.transform([review])
    review_vectorized_nb = nb_vectorizer.transform([review])

    lr_pred = logreg_model.predict(review_vectorized_lr)[0]
    nb_pred = nb_model.predict(review_vectorized_nb)[0]

    results = {
        "Logistic Regression": "Positive" if lr_pred == 1 else "Negative",
        "Multinomial NB": "Positive" if nb_pred == 1 else "Negative"
    }

    return results

if __name__ == "__main__":
    print("\n🎬 IMDb Sentiment Analysis Tool")
    print("Type 'exit' to quit.\n")

    while True:
        review = input("Enter a movie review: ")
        if review.lower() == 'exit':
            break
        output = predict_sentiment(review)
        print(f"Results → Logistic Regression: {output['Logistic Regression']} | Naive Bayes: {output['Multinomial NB']}\n")
