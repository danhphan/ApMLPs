import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import config, dataset, engine

from torch.utils.data import DataLoader
from model import BertBaseUncased
from sklearn import model_selection, metrics
from transformers import get_linear_schedule_with_warmup

torch.cuda.empty_cache()

def train():
    # Get data
    dfx = pd.read_csv(config.TRAINING_FILE).fillna("none")
    dfx.sentiment = dfx.sentiment.apply(lambda x : 1 if x == "positive" else 0)
    df_train, df_valid = model_selection.train_test_split(dfx, test_size=0.1, 
        random_state=42, stratify=dfx.sentiment.values)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    # Get dataset and dataloader
    train_ds = dataset.BERTDataset(review=df_train.review.values, target=df_train.sentiment.values)
    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, num_workers=8)
    valid_ds = dataset.BERTDataset(review=df_valid.review.values, target=df_valid.sentiment.values)
    valid_dl = DataLoader(valid_ds, batch_size=config.BATCH_SIZE, num_workers=8)
    # device and model
    device = torch.device("cuda")
    model = BertBaseUncased()
    model.to(device)
    # Create paramaters which needed to optimize
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.001,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }
    ]
    # Calculate training steps used by scheduler
    num_train_steps = int(len(df_train) / config.BATCH_SIZE * config.EPOCHS)

    #optimizer = Adam(optimizer_parameters, lr=3e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, 
        num_training_steps=num_train_steps)
    model = nn.DataParallel(model)

    print("Start training")
    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_dl, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_dl, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"Epoch={epoch}, Accuracy={accuracy}")

if __name__=="__main__":
    train()

