import os
import pandas as pd
import numpy as np
import albumentations
import torch
from torch.utils.data import DataLoader

from sklearn import metrics
from sklearn.model_selection import train_test_split

import dataset
import engine
from model import get_model

if __name__ == "__main__":
    
    # Read data
    data_path = "../data/"
    df = pd.read_csv(os.path.join(data_path, "train.csv"))
    # Get list of image files
    images = df.ImageId.values.tolist()
    images = [os.path.join(data_path, "train_png", i + ".png") for i in images]
    # Set device and epochs
    device = "cuda"
    epochs = 10

    targets =df.target.values
    model = get_model(pretrained=False)
    model.to(device)

    # mean and std of RGB channels
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    aug = albumentations.Compose(
        [albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)]
    )

    train_images, valid_images, train_targets, valid_targets = train_test_split(
        images, targets, stratify=targets, random_state=42
    )

    train_dataset = dataset.ClassificationDataset(
        image_paths=train_images, targets=train_targets, 
        resize=(227, 227), augmentations=aug
    )
    valid_dataset = dataset.ClassificationDataset(
        image_paths=train_images, targets=train_targets,
        resize=(227, 227), augmentations=aug
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    for epoch in range(epochs):
        engine.train(train_loader, model, optimizer, device)
        preds, valid_targets = engine.evaluate(valid_loader, model, device)
        auc = metrics.roc_auc_score(valid_targets, preds)
        print(f"Epoch={epoch}, Valid AUC={auc}")





    



