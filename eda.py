import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Spotify_Youtube.csv', index_col=0)

# drop unnecessary columns
df = df.drop(['Url_spotify', 'Url_youtube', 'Description', 'Channel'], axis=1)

# missing values
print(df.nunique())
df.info()
print(df.isnull().sum())
df = df.dropna()



# replace string with int
df['Licensed'] = df['Licensed'].astype(int)
df['official_video'] = df['official_video'].astype(int)
df['Album_type'] = df['Album_type'].replace('album', 0)
df['Album_type'] = df['Album_type'].replace('single', 1)
df['Album_type'] = df['Album_type'].replace('compilation', 2)
df.info()
#drop duplicates
df = df.sort_values('Stream', ascending=False)
df = df.drop_duplicates(subset='Uri', keep='first')

df.info()

print(df.nunique())
# correlation
df_corr = df.drop(['Artist', 'Track', 'Album', 'Album_type', 'Title', 'Key', 'Uri'], axis=1)
df_corr.rename(columns={'Danceability':'Taneczność'}, inplace=True)
corrM = df_corr.corr(method='pearson')
plt.figure(figsize=(20, 9))
heatmap = sns.heatmap(corrM, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Macierz korelacji')
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
