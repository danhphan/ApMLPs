import io
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn import linear_model, metrics, model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

def load_vectors(fname):
    # Taken from: fasttext.cc/docs/en/english-vectors.html
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def sentence2vec(s, embedding_dict, stop_words, tokenizer):
    """
    Given a sentence and other information,
    this function returns embedding for the whole sentence
    :param s: sentence, string
    :param embedding_dict: dictionary word: vector
    :param stop_words: list of stop words
    :param tokenizer: a tokeniztion function
    :return: embedding of the sentence
    """
    # Process sentence into words
    words = str(s).lower()
    words = tokenizer(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    # Create and store embedding
    M = []
    for w in words:
        if w in embedding_dict:
            M.append(embedding_dict[w])
    # Return zeros if no vectors found
    if len(M) == 0: return np.zeros(300)
    # Sum and normalized vector
    M = np.array(M)
    v = M.sum(axis=0) 
    v = v/np.sqrt((v ** 2).sum())
    return v


if __name__ == "__main__":
    df = pd.read_csv("../data/imdb.csv")
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)
    df = df.sample(frac=1).reset_index(drop=True)
    # Loading embeddings into def load_vectors(fname):)
    #Fetch labels
    y = df.sentiment.values
    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold_, (t_, v_) in enumerate(kf.split(X=vectors, y=y)):
        print(f"Training fold: {fold_}")
        xtrain = vectors[t_, :]
        ytrain = y[t_]
        xvalid = vectors[v_, :]
        yvalid = y[v_]

        model = linear_model.LogisticRegression()
        model.fit(xtrain, ytrain)
        preds = model.predict(xvalid)
        acc = metrics.accuracy_score(yvalid, preds)
        print(f"Accuracy = {acc}")

