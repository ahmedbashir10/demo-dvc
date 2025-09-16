import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

processed_path = Path(params["paths"]["processed"])
train_path = Path(params["paths"]["train"])
test_path = Path(params["paths"]["test"])

test_size = float(params["train"]["test_size"])
shuffle = bool(params["train"]["shuffle"])
random_state = int(params["train"]["random_state"])

df = pd.read_csv(processed_path)

X = df[["idx"]].values
y = df["Price"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
)

train_df = pd.DataFrame({"idx": X_train.flatten(), "Price": y_train})
test_df = pd.DataFrame({"idx": X_test.flatten(), "Price": y_test})

train_path.parent.mkdir(parents=True, exist_ok=True)
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Split done -> {train_path}, {test_path}")
