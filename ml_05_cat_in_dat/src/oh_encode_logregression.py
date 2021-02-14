import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing


def run(fold):
    df = pd.read_csv("../input/cat_train_folds.csv")

    features = [f for f in df.columns if f not in ["id","target","kfold"]]

    # Fill NA with NONE
    for col in features:
        df.loc[:,col] = df[col].fillna("NONE").astype(str)
    # Get training and validation sets using kfolds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    onehot_encoder = preprocessing.OneHotEncoder()
    # Fit onehot encoding for both training and validation set
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
    onehot_encoder.fit(full_data[features])
    # Transform
    x_train = onehot_encoder.transform(df_train[features])
    x_valid = onehot_encoder.transform(df_valid[features])
    y_train = df_train.target.values
    y_valid = df_valid.target.values

    # Create a logistic regression model
    model = linear_model.LogisticRegression()
    model.fit(x_train, y_train)
    # Predict
    valid_preds = model.predict_proba(x_valid)[:,1]
    # Get roc auc score
    auc = metrics.roc_auc_score(y_valid, valid_preds)
    print(f"Fold: {fold}, AUC: {auc}")


if __name__ == "__main__":
    for f in range(5):
        run(fold=f)