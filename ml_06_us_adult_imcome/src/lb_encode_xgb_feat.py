import itertools
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, metrics

def feature_engineering(df, cat_cols):
    """Generate 2-combinations for categorical features"""
    combi = list(itertools.combinations(cat_cols,2))
    for c1, c2 in combi:
        df.loc[:,c1+"_"+c2] = df[c1].astype(str) + "_" + df[c2].astype(str)
    return df

def run(fold):
    df = pd.read_csv("../input/adult_folds.csv")
    target = "income"
    num_cols = ["age", "fnlwgt", "capital.gain", "capital.loss", "hours.per.week"]
    cat_cols = [c for c in df.columns if c not in num_cols and c not in ("kfold", target)]

    # Add new features
    df = feature_engineering(df, cat_cols)

    features = [f for f in df.columns if f not in ("kfold", target)]
    # Fill NA with NONE for categorical features
    for c in features:
        if c not in num_cols:
            df.loc[:,c] = df[c].astype(str).fillna("NONE")
    # Label encoding
    for col in features:
        if col not in num_cols:
            lb_encoder = preprocessing.LabelEncoder()
            df.loc[:,col] = lb_encoder.fit_transform(df[col])
    # Get training and validation data
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    x_train = df_train[features].values
    y_train = df_train[target].values
    x_valid = df_valid[features].values
    y_valid = df_valid[target].values
    # Fit model
    model_xgb = xgb.XGBClassifier(n_jobs=-1, eval_metric="logloss", max_depth=7)
    model_xgb.fit(x_train,y_train)
    # Predict
    valid_preds = model_xgb.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(y_valid, valid_preds)
    print(f"Fold {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold in range(5):
        run(fold)



