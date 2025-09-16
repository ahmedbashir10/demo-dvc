import json
import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

with open("params.yaml") as f:
    params = yaml.safe_load(f)

test_path = Path(params["paths"]["test"])
model_path = Path(params["paths"]["model"])
metrics_path = Path(params["paths"]["metrics"])
preds_path = Path(params["paths"]["preds"])
preds_path.parent.mkdir(parents=True, exist_ok=True)

test_df = pd.read_csv(test_path)
X_test = test_df[["idx"]].values
y_test = test_df["Price"].values

with open(model_path, "rb") as f:
    model, poly = pickle.load(f)

X_test_poly = poly.transform(X_test)
y_pred = model.predict(X_test_poly)

mae = float(mean_absolute_error(y_test, y_pred))
rmse = float(np.sqrt(((y_test - y_pred) ** 2).mean()))

metrics = {"MAE": mae, "RMSE": rmse, "n_test": int(len(y_test))}
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

preds_df = pd.DataFrame({
    "idx": test_df["idx"].values,
    "y_true": y_test,
    "y_pred": y_pred
})
preds_df.to_csv(preds_path, index=False)

print(f"Evaluated -> {metrics_path}, {preds_path}")
