import copy
import pandas as pd
from sklearn import preprocessing, metrics
import xgboost as xgb

def mean_target_encoding(data):
    # Make a copy of dataframe
    df = copy.deepcopy(data)

    target_mapping = {"<=50K":0, ">50K":1}
    df.loc[:,"income"] = df["income"].map(target_mapping)

    num_cols = ["age", "fnlwgt", "capital.gain", "capital.loss", "hours.per.week"]
    features = [f for f in df.columns if f not in num_cols and f not in ("kfold","income")]
    # Fill NA with NONE for categorical features
    for c in features:
        if c not in num_cols:
            df.loc[:,c] = df[c].astype(str).fillna("NONE")
    # Label encoding
    for cat in features:
        lb_encoder = preprocessing.LabelEncoder()
        df.loc[:,cat] = lb_encoder.fit_transform(df[cat])

    # A list to store 5 validation dataframes
    encoded_dfs = []
    for fold in range(5):
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        for col in features:
            if col not in num_cols:
                mapping_dict = dict(df_train.groupby(col)["income"].mean())
                df_valid.loc[:, col+"_enc"] = df_valid[col].map(mapping_dict)
        # Store encoded dataframe
        encoded_dfs.append(df_valid)
    # Create full data frame again
    encoded_data = pd.concat(encoded_dfs, axis=0)
    return encoded_data


def run(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    features = [f for f in df.columns if f not in ("kfold","income")]

    x_train = df_train[features].values
    y_train = df_train["income"].values
    x_valid = df_valid[features].values
    y_valid = df_valid["income"].values

    model_xgb = xgb.XGBClassifier(n_jobs=-1, max_depth=7, eval_metric="logloss")
    model_xgb.fit(x_train, y_train)

    valid_preds = model_xgb.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(y_valid, valid_preds)
    print(f"Fold {fold}, AUC = {auc}")

if __name__ == "__main__":
    df = pd.read_csv("../input/adult_folds.csv")
    # Create target encoded categories and munge data
    df = mean_target_encoding(df)
    # Cross validation
    for fold_ in range(5):
        run(df, fold_)



            