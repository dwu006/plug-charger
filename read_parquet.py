import pandas as pd
import numpy as np

df = pd.read_parquet('test/data/chunk-000/test.parquet')
print("Columns:", df.columns.tolist())
print("\nShape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nColumn info:")
print(df.info())