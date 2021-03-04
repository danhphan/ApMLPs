import pandas as pd
from sklearn import model_selection

df = pd.read_csv("../data/imdb.csv")
# Map positive to 1, and negative to 0
df.sentiment = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)

df["kfold"] = -1
df = df.sample(frac=1).reset_index(drop=True)
y = df.sentiment.values
skf = model_selection.StratifiedKFold(n_splits=5)
for f, (t_, v_) in enumerate(skf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f


df.to_csv("../data/imdb_folds.csv", index=False)