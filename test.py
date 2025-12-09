import pandas as pd

df = pd.read_csv("synthetic_trips.csv")
print(df.columns.tolist())
print(df.head())
