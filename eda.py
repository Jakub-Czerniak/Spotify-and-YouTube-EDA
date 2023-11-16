import numpy as np
import pandas as pd

df = pd.read_csv('Spotify_Youtube.csv')

print(df.info())
print(df.head())
print(df.describe())
