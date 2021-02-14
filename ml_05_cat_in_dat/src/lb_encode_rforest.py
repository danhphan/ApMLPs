import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

def run(fold):

    df = pd.read_csv("../input/cat_train_folds.csv")

    features = [f for f in df.columns if f not in ["id","target","kfold"]]

    # Fill NA with NONE
    for col in features:
        df.loc[:,col] = df[col].astype(str).fillna("NONE")

    for col in features:
        # Label encoding
        lbl_encoder = preprocessing.LabelEncoder()
        lbl_encoder.fit(df[col])
        df.loc[:,col] = lbl_encoder.transform(df[col])
    # Get training and validation sets
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values
    y_train = df_train.target.values
    y_valid = df_valid.target.values
    # Create a random forest model and fit training data
    model = ensemble.RandomForestClassifier(n_jobs=-1)
    model.fit(x_train,y_train)
    # Predict
    valid_preds = model.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(y_valid, valid_preds)
    print(f"Fold: {fold}, AUC: {auc}")

if __name__ == "__main__":
    for f in range(5):
        run(fold=f)


