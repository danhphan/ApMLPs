import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, metrics, ensemble



def run(fold):
    df = pd.read_csv("../input/cat_train_folds.csv")

    features = [f for f in df.columns if f not in ("id","target","kfold")]
    # Fill NA with NONE
    for col in features:
        df.loc[:,col] = df[col].astype(str).fillna("NONE")

    # Label encoding
    for col in features:
        lbl_encoder = preprocessing.LabelEncoder()
        df.loc[:,col] = lbl_encoder.fit_transform(df[col])

    x_train = df.loc[df.kfold != fold, features].values
    x_valid = df.loc[df.kfold == fold, features].values
    y_train = df.loc[df.kfold != fold].target.values
    y_valid = df.loc[df.kfold == fold].target.values

    # Create xgb model and fit training data
    # model = ensemble.GradientBoostingClassifier()
    model = xgb.XGBClassifier(n_estimators=100, n_jobs=-1, max_depth=7)
    model.fit(x_train, y_train)

    # Predict
    valid_preds = model.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(y_valid, valid_preds)
    print(f"Fold: {fold}, AUC: {auc}")

if __name__ == "__main__":
    for fold in range(5):
        run(fold)

