import pandas as pd

df = pd.read_csv("data/passwords.csv")

# Splitting the DataFrame into train and test sets
n_rows = len(df) // 2
train_df = df.loc[:n_rows]
test = df.loc[n_rows:]

# You can now use train_df and test_df for your model training and testing
train_df.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
