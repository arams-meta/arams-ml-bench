import os
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

train = pd.read_csv("/app/data/train.csv")
test = pd.read_csv("/app/data/test.csv")

model = LinearRegression()
model.fit(train[["x"]], train["y"])

predictions = model.predict(test[["x"]])

os.makedirs("/output", exist_ok=True)
output = pd.DataFrame({"id": range(len(predictions)), "prediction": predictions})
output.to_csv("/output/predictions.csv", index=False)

with open("/output/model.pkl", "wb") as f:
    pickle.dump(model, f)
