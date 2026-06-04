import pandas as pd
import numpy as np

df = pd.read_csv("data/passwords_raw.csv")
df["Times"] = np.log10(df["Times"])
df.to_csv("data/passwords.csv", index=False)
