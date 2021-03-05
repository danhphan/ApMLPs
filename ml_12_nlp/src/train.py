import io
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import tensorflow as tf
from sklearn import metrics

import config, dataset, engine, lstm

def load_vectors(fname):
    # Taken from: fasttext.cc/docs/en/english-vectors.html
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def create_embedding_matrix(word_index, embedding_dict):
    """
    This function creates the embedding matrix
    :param word_index: a dictionary with word:index_value
    :param embedding_dict: a dictionary with word:embedding_vector
    :return: a numpy array with embedding vectors for all known words
    """
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]
    return embedding_matrix

def run(df, fold):
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    print("Fitting tokenizer")
    # Use tokenizer from tf.keras
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())
    # Convert data to sequences
    xtrain = tokenizer.texts_to_sequences(train_df.review.values)
    xvalid = tokenizer.texts_to_sequences(valid_df.review.values)
    # Zero pad given the maximum length
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=config.MAX_LEN)
    xvalid = tf.keras.preprocessing.sequence.pad_sequences(xvalid, maxlen=config.MAX_LEN)
    # Generate dataset and dataloader
    train_ds = dataset.ImdbDataset(reviews=xtrain, targets=train_df.sentiment.values)
    train_dl = DataLoader(train_ds, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4)
    valid_ds = dataset.ImdbDataset(reviews=xvalid, targets=valid_df.sentiment.values)
    valid_dl = DataLoader(valid_ds, batch_size=config.VALID_BATCH_SIXE, num_workers=4)
    print("Loading embedding")
    embedding_dict = load_vectors("../data/crawl-300d-2M.vec")
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_dict)
    # Create device, and model
    device = torch.device("cuda")
    model = lstm.LSTM(embedding_matrix)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Training model")
    best_accuracy = 0
    early_stop_counter = 0
    for epoch in range(config.EPOCHS):
        engine.train(train_dl, model, optimizer, device)
        outputs, targets = engine.evaluate(valid_dl, model, device)
        # using 0.5 threshold after sigmoid
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)        
        # Store best accuracy
        if accuracy > best_accuracy: 
            best_accuracy = accuracy
            print(f"Fold={fold}, Epoch={epoch}, Accuracy={accuracy}")
        else: 
            early_stop_counter += 1
        if early_stop_counter > 2: 
            break

if __name__ == "__main__":
    df = pd.read_csv("../data/imdb_folds.csv")
    for fold_ in range(5):
        run(df, fold_)
