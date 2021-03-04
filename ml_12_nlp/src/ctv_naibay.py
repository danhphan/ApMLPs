import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn import naive_bayes, metrics
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("../data/imdb_folds.csv")

for fold_ in range(5):

    train_df = df[df.kfold != fold_].reset_index(drop=True)
    valid_df = df[df.kfold == fold_].reset_index(drop=True)

    # Countvectorize reviews
    count_vec = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)
    count_vec.fit(train_df.review)

    xtrain = count_vec.transform(train_df.review)
    xvalid = count_vec.transform(valid_df.review)

    # Model
    model = naive_bayes.MultinomialNB()
    model.fit(xtrain, train_df.sentiment)

    # Predict
    preds = model.predict(xvalid)
    acc = metrics.accuracy_score(valid_df.sentiment, preds)

    print(f"Fold={fold_}, Accuracy={acc}")

