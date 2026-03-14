import pandas as pd
import numpy as np

df = pd.read_parquet('test/data/chunk-000/test.parquet')
print(df.columns.tolist())
print(df.shape)
print(df.head())