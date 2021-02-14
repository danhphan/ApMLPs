import pandas as pd
from sklearn import model_selection

df_train = pd.read_csv("../input/cat_train.csv")
df_train["kfold"] = -1

# Shuffling data
df_train = df_train.sample(frac=1).reset_index(drop=True)

y = df_train.target.values

kf = model_selection.StratifiedKFold(n_splits=5)

for f, (t_, v_) in enumerate(kf.split(X=df_train,y=y)):
    df_train.loc[v_, 'kfold'] = f

df_train.to_csv("../input/cat_train_folds.csv", index=False)