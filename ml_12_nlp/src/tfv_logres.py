import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn import linear_model, metrics, model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("../data/imdb_folds.csv")

for fold_ in range(5):
    train_df = df[df.kfold != fold_].reset_index(drop=True)
    valid_df = df[df.kfold == fold_].reset_index(drop=True)

    # TF-IDF vectorize reviews: with and without n-grams
    # tfidf_vec = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
    # With trigrams
    tfidf_vec = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None, ngram_range=(1,2))
    tfidf_vec.fit(train_df.review)

    xtrain = tfidf_vec.transform(train_df.review)
    xvalid = tfidf_vec.transform(valid_df.review)

    # Model
    model = linear_model.LogisticRegression()
    model.fit(xtrain, train_df.sentiment)

    # Predict
    preds = model.predict(xvalid)
    acc = metrics.accuracy_score(valid_df.sentiment, preds)

    print(f"Fold={fold_}, Accuracy={acc}")
