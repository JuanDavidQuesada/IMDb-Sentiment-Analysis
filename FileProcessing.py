import os
import re
import nltk
import swifter
import pandas as pd
import fastparquet


def clean_preprocess(text):

    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    words = text.split()
    words = [word for word in words if word not in nltk.corpus.stopwords.words("english")]
    text = " ".join(words)

    text = re.sub(r"\s+", " ", text)

    return text

if __name__ == "__main__":

    train_dir = "aclImdb/train"
    test_dir = "aclImdb/test"

    train_pos_dir = os.path.join(train_dir, "pos")
    train_neg_dir = os.path.join(train_dir, "neg")

    test_pos_dir = os.path.join(test_dir, "pos")
    test_neg_dir = os.path.join(test_dir, "neg")

    train_reviews =[]
    train_sentiments = []
    test_reviews = []
    test_sentiments = []

    for filename in os.listdir(train_pos_dir):
        with open(os.path.join(train_pos_dir, filename), "r", encoding = "utf-8") as f:
            train_reviews.append(f.read())
            train_sentiments.append(1)

    for filename in os.listdir(train_neg_dir):
        with open(os.path.join(train_neg_dir, filename), "r", encoding = "utf-8") as f:
            train_reviews.append(f.read())
            train_sentiments.append(0)

    for filename in os.listdir(test_pos_dir):
        with open(os.path.join(test_pos_dir, filename), "r", encoding = "utf-8") as f:
            test_reviews.append(f.read())
            test_sentiments.append(1)

    for filename in os.listdir(test_neg_dir):
        with open(os.path.join(test_neg_dir, filename), "r", encoding = "utf-8") as f:
            test_reviews.append(f.read())
            test_sentiments.append(0)


    train_df = pd.DataFrame({"review": train_reviews, "sentiment": train_sentiments})
    test_df = pd.DataFrame({"review": test_reviews, "sentiment": test_sentiments})


    print(train_df.head())



    print("Cleaning training data")
    train_df['review'] = train_df['review'].swifter.apply(clean_preprocess)
    print("Done")

    print("Cleaning testing data")
    test_df['review'] = test_df['review'].swifter.apply(clean_preprocess)
    print("Done")

    print("Parsing training data to parquet")
    train_df.to_parquet("train.parquet")
    print("Done")

    print("Parsing testing data to parquet")
    test_df.to_parquet("test.parquet")
    print("Done")
