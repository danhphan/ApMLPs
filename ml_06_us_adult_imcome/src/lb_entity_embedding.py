import os, gc, joblib
import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from tensorflow.keras import layers, optimizers, callbacks, utils
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K

# Set keras to use CPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

def create_model(data, cat_cols):
    inputs = [] # init list of inputs for embeddings
    outputs = [] # Init list of outputs for embeddings

    for c in cat_cols:
        num_unique_values = int(data[c].nunique())
        embed_dim = int(min(np.ceil(num_unique_values/2), 50))
        # simple keras input layer with size 1
        inp = layers.Input(shape=(1,))
        # add embedding layer to raw input
        out = layers.Embedding(num_unique_values + 1, embed_dim, name=c)(inp)
        # 1-d spatial dropout
        out = layers.SpatialDropout1D(0.3)(out)
        # reshape input to the dimession of embedding
        out = layers.Reshape(target_shape=(embed_dim, ))(out)
        # Add to list inputs and outputs
        inputs.append(inp)
        outputs.append(out)

    # Concat and add a batchnorm layer
    x = layers.Concatenate()(outputs)
    x = layers.BatchNormalization()(x)
    # Add more dense layers with dropout
    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    
    # use softmax for output
    y = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=y)
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model

def run(fold):
    df = pd.read_csv("../input/adult_folds.csv")
    num_cols = ["age", "fnlwgt", "capital.gain", "capital.loss", "hours.per.week"]
    cat_cols = [c for c in df.columns if c not in num_cols and c not in ("id","kfold", "income")]
    features = [f for f in df.columns if f not in ("id","kfold", "income")]
    # Fill NA with NONE, and label encoding
    for c in features:
        if c not in num_cols:
            df.loc[:,c] = df[c].astype(str).fillna("NONE")
            lb_encoder = preprocessing.LabelEncoder()
            df.loc[:,c] = lb_encoder.fit_transform(df[c])
    # Get training and validation data
    df_train = df[df.kfold != fold].reset_index(drop=True)    
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # Create tf.keras model
    model = create_model(df, cat_cols)

    # Our features are list of list
    x_train = [df_train[features].values[:, k] for k in range(len(features))]
    y_train = df_train["income"].values
    x_valid = [df_valid[features].values[:, k] for k in range(len(features))]
    y_valid = df_valid["income"].values

    # Fit model
    model.fit(x_train, y_train, validation_data=(x_valid, y_valid), verbose=1, batch_size=64, epochs=3)
    valid_preds = model.predict(x_valid)[:, 1]
    auc = metrics.roc_auc_score(y_valid, valid_preds)
    print(f"Fold {fold}, AUC = {auc}")
    K.clear_session()

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)



    
    