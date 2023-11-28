import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Spotify_Youtube.csv', index_col=0)

# drop unnecessary columns
df = df.drop(['Url_spotify', 'Uri', 'Url_youtube', 'Description', 'Channel'], axis=1)

# missing values
df.info()
df.isna().sum()
df = df.dropna()
df.info()
df.isna().sum()

# replace string with int
df['Licensed'] = df['Licensed'].astype(int)
df['official_video'] = df['official_video'].astype(int)
df['Album_type'] = df['Album_type'].replace('album', 0)
df['Album_type'] = df['Album_type'].replace('single', 1)
df['Album_type'] = df['Album_type'].replace('compilation', 2)
print(df.Album_type.unique())
# correlation
df_corr = df.drop(['Artist', 'Track', 'Album', 'Album_type', 'Title'], axis=1)
corrM = df_corr.corr(method='pearson')
print(corrM)
plt.figure(figsize=(20, 9))
heatmap = sns.heatmap(corrM, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Macierz korelacji')
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
