import pandas as pd

df = pd.read_parquet('test/data/chunk-000/test.parquet')
df.to_csv('test.csv', index=False)

