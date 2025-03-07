import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("ai4i2020.csv")
print(df.dtypes)
print(df.head(0 ))
df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
