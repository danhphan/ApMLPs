import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, metrics

def run(fold):
    df = pd.read_csv("../input/adult_folds.csv")

    target = "income"
    num_cols = ["age", "fnlwgt", "capital.gain", "capital.loss", "hours.per.week"]
    features = [f for f in df.columns if f not in ("kfold",target)]
   
    mapping = {"<=50K" : 0, ">50K" : 1}
    df.loc[:, target] = df[target].map(mapping)

    # Fill NA with NONE
    for col in features:
        if col not in num_cols:        
            df.loc[:,col] = df[col].astype(str).fillna("NONE")

    # Label encoding
    for col in features:
        if col not in num_cols:
            lb_encoder = preprocessing.LabelEncoder()
            df.loc[:, col] = lb_encoder.fit_transform(df[col])

    # Get training, validation data
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values
    y_train = df_train[target].values
    y_valid = df_valid[target].values

    # Create model and fit training data
    model = xgb.XGBClassifier(n_jobs=-1, n_estimators=200, max_depth=7, eval_metric="logloss")
    model.fit(x_train, y_train)
    
    # Predict
    valid_preds = model.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(y_valid, valid_preds)
    print(f"Fold: {fold}, AUC: {auc}")

if __name__ == "__main__":
    for fold in range(5):
        run(fold)