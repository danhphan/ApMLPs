import pandas as pd
from sklearn import preprocessing, metrics, linear_model



def run(fold):
    df = pd.read_csv("../input/adult_folds.csv")

    target = "income"
    num_cols = ["age", "fnlwgt", "capital.gain", "capital.loss", "hours.per.week"]
    features = [f for f in df.columns if f not in ("kfold",target)]

    # Fill NA with NONE
    for col in features:
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # Normalise numeric columns
    # df.loc[:, num_cols] = df[num_cols]/df[num_cols].max()

    # Get training and validation data
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # One hot encoding
    oh_encoder = preprocessing.OneHotEncoder()
    oh_encoder.fit(df[features])

    x_train = oh_encoder.transform(df_train[features])
    x_valid = oh_encoder.transform(df_valid[features])
    y_train = df_train[target].values
    y_valid = df_valid[target].values

    # Create model and fit training data
    model = linear_model.LogisticRegression()
    model.fit(x_train, y_train)

    # Predict
    valid_preds = model.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(y_valid, valid_preds)

    print(f"Fold: {fold}, AUC: {auc}")


if __name__ == "__main__":
    for fold in range(5):
        run(fold)