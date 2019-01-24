import pandas as pd

df = pd.read_csv('data.csv')
df.to_latex('data.tex', index=False)