import os
import numpy as np
import pandas as pd

from PIL import Image
from sklearn import ensemble, metrics, model_selection
from tqdm import tqdm

def create_dataset(train_df, image_dir):
    """
    This function takes the training dataframe and ouputs traning array and labels
    :param train_df: dataframe with ImageId, Target columns
    :param image_dir: image directory, string
    :return: X, y (training array with features and labels)
    """
    images = []
    targets = []
    # Loop over the dataframe
    for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc="processing images"):
        img_id = row["ImageId"]
        img_path = os.path.join(image_dir, img_id)
        # Open image and resize
        image = Image.open(img_path + ".png")
        image = image.resize((256,256), resample=Image.BILINEAR)
        # Convert image to list
        image = np.array(image).ravel()
        # Append to images and tartgets lists
        images.append(image)
        targets.append(int(row["target"]))
    
    # Convert list of list of images to numpy array
    images = np.array(images)
    print(images.shape)
    return images, targets


if __name__ == "__main__":
    csv_path = "../data/train.csv"
    img_path = "../data/train_png/"

    df = pd.read_csv(csv_path)
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    # Initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    for fold_ in range(5):
        # Create temporary dataframes for train and validation
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        valid_df = df[df.kfold == fold_].reset_index(drop=True)
        # Create train and validation datasets
        xtrain, ytrain = create_dataset(train_df, img_path)
        xvalid, yvalid = create_dataset(valid_df, img_path)
        # Fit random forest
        clf = ensemble.RandomForestClassifier(n_jobs=-1)
        clf.fit(xtrain, ytrain)
        # Predict
        preds = clf.predict_proba(xvalid)[:, 1]
        auc = metrics.roc_auc_score(yvalid, preds)
        print(f"Fold={fold_}, AUC={auc}")



