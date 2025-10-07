
import fastparquet
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from FileProcessing import clean_preprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, mean_absolute_error
import pickle

def save_to_pickle(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    train_df = pd.read_parquet("train.parquet")
    test_df = pd.read_parquet("test.parquet")

    train_X = train_df["review"]
    train_y = train_df["sentiment"]

    test_X = test_df["review"]
    test_y = test_df["sentiment"]

    vectorizer = CountVectorizer()
    train_X = vectorizer.fit_transform(train_X)
    test_X = vectorizer.transform(test_X)


    model = MultinomialNB()
    print("Training model")
    model.fit(train_X, train_y)
    print("Done")
    save_to_pickle("MultinomialNB_model.pkl", model)
    save_to_pickle("MultinomialNB_vectorizer.pkl", vectorizer)

def test_model():

    print("Testing model")
    predictions = model.predict(test_X)
    print("Done")
    return predictions

def predict_sentiment(text, vectorizer, model):

    clean_text = clean_preprocess(text)

    vectorized_text = vectorizer.transform([clean_text])

    prediction = model.predict(vectorized_text)

    return prediction[0]


def metrics():

    predictions = model.predict(test_X)
    acc = accuracy_score(test_y, predictions)
    prec = precision_score(test_y, predictions)
    rec = recall_score(test_y, predictions)
    f1 = f1_score(test_y, predictions)
    cm = confusion_matrix(test_y, predictions)
    report = classification_report(test_y, predictions, target_names=['Negative', 'Positive'])

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)


