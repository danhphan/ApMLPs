import pandas as pd
from sklearn import preprocessing

# Read data
df_train = pd.read_csv("../input/cat_train.csv")
df_test = pd.read_csv("../input/cat_test.csv")

# Create a fake target column
test.loc[:,"target"] = -1

# Concat train and test data
data = pd.concat([df_train, df_test]).reset_index(drop=True)

# Make a list of interested feature
features = [x for x in train.columns if x not in ["id", "target"]]

for feat in features:
    lbl_encoder = preprocessing.LabelEncoder()
    # Fill NA and convert to string
    temp = data[feat].fillna("NONE").astype(str).values
    data.loc[:,feat] = lbl_encoder.fit_transform(temp)

# Split rain and test again
train = data[data.target != -1].reset_index(drop=True)
test = data[data.target == -1].reset_index(drop=True)


