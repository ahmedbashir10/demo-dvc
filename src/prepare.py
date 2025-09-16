import pandas as pd
from pathlib import Path
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

raw_path = Path(params["paths"]["raw_data"])
processed_path = Path(params["paths"]["processed"])

df = pd.read_csv(raw_path)

# Ensure numeric 'Price' column
df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",", "", regex=True))

# Add sequential index
df = df.reset_index(drop=True)
df["idx"] = df.index

processed_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(processed_path, index=False)

print(f"Prepared data -> {processed_path}")
