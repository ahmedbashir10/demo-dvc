import pickle
import yaml
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

with open("params.yaml") as f:
    params = yaml.safe_load(f)

deg = int(params["train"]["degree"])

train_path = Path(params["paths"]["train"])
model_path = Path(params["paths"]["model"])
model_path.parent.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(train_path)
X = train_df[["idx"]].values
y = train_df["Price"].values

poly = PolynomialFeatures(degree=deg)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

with open(model_path, "wb") as f:
    pickle.dump((model, poly), f)

print(f"Trained degree={deg} -> {model_path}")
