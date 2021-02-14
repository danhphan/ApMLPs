import pandas as pd
from sklearn import ensemble, metrics, preprocessing, decomposition
from scipy import sparse

def run(fold):
    df = pd.read_csv("../input/cat_train_folds.csv")

    features = [f for f in df.columns if f not in ["id","target","kfold"]]

    for col in features:
        df.loc[:, col]  = df[col].astype(str).fillna("NONE")
    
    df_train = df[df.kfold != fold].reset_index(drop=True)    
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # One-hot encoding
    data_full = pd.concat([df_train[features], df_valid[features]], axis=0)
    oh_encoder = preprocessing.OneHotEncoder()
    oh_encoder.fit(data_full)
    # Transform
    x_train = oh_encoder.transform(df_train[features])
    x_valid = oh_encoder.transform(df_valid[features])
    y_train = df_train.target.values
    y_valid = df_valid.target.values

    # SVD to reduce data to 120 components
    svd = decomposition.TruncatedSVD(n_components=120)
    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)

    # transform to sparse data
    x_train = svd.transform(x_train)
    x_valid = svd.transform(x_valid)

    # Create model and fit training data
    model = ensemble.RandomForestClassifier(n_jobs=-1)
    model.fit(x_train, y_train)
    # Predict
    valid_preds = model.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(y_valid, valid_preds)
    print(f"Fold: {fold}, AUC: {auc}")

if __name__ == "__main__":
    for f in range(5):
        run(fold=f)
