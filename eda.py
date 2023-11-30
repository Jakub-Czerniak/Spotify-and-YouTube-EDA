import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

df = pd.read_csv('Spotify_Youtube.csv', index_col=0)

# drop unnecessary columns
df = df.drop(['Url_spotify', 'Url_youtube', 'Description', 'Channel'], axis=1)
df_start = df.copy()
df_start.rename(columns={'Danceability':'Taneczność', 'Energy':'Energia', 'Loudness':'Głośność', 'Speechiness':'Tekstowość', 'Acousticness':'Akustyczność', 'Instrumentalness':'Instrumentalność', 'Liveness':'Żywość', 'Valence':'Pozytywność', 'Duration_ms':'Czas trwania', 'Stream':'Odsłuchania', 'Views':'Wyświetlenia', 'Likes':'Polubienia', 'Comments':'Komentarze', 'Licensed':'Licencjonowany', 'official_video':'Oficjalny film'}, inplace=True)

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
#df = df.sort_values('Stream', ascending=False)
df = df.drop_duplicates(subset='Uri', keep='first')


df.info()

print(df.nunique())
# correlation
df_corr = df.drop(['Artist', 'Track', 'Album', 'Album_type', 'Title', 'Key', 'Uri'], axis=1)
df_corr.rename(columns={'Danceability':'Taneczność', 'Energy':'Energia', 'Loudness':'Głośność', 'Speechiness':'Tekstowość', 'Acousticness':'Akustyczność', 'Instrumentalness':'Instrumentalność', 'Liveness':'Żywość', 'Valence':'Pozytywność', 'Duration_ms':'Czas trwania', 'Stream':'Odsłuchania', 'Views':'Wyświetlenia', 'Likes':'Polubienia', 'Comments':'Komentarze', 'Licensed':'Licencjonowany', 'official_video':'Oficjalny film'}, inplace=True)
corrM = df_corr.corr(method='spearman')
plt.figure(figsize=(12, 15))
heatmap = sns.heatmap(corrM, vmin=-1, vmax=1, annot=True, cmap='BrBG', cbar=False)
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')

# relations
fig, ax =plt.subplots(3,3,figsize=(12,10))
sns.scatterplot(y='Odsłuchania', x='Taneczność', data=df_start, ax=ax[0][0])
sns.scatterplot(y='Odsłuchania', x='Energia', data=df_start, ax=ax[0][1])
sns.scatterplot(y='Odsłuchania', x='Głośność', data=df_start, ax=ax[0][2])
sns.scatterplot(y='Odsłuchania', x='Tekstowość', data=df_start, ax=ax[1][0])
sns.scatterplot(y='Odsłuchania', x='Akustyczność', data=df_start, ax=ax[1][1])
sns.scatterplot(y='Odsłuchania', x='Instrumentalność', data=df_start, ax=ax[1][2])
sns.scatterplot(y='Odsłuchania', x='Żywość', data=df_start, ax=ax[2][0])
sns.scatterplot(y='Odsłuchania', x='Pozytywność', data=df_start, ax=ax[2][1])
sns.scatterplot(y='Odsłuchania', x='Czas trwania', data=df_start, ax=ax[2][2])
fig.savefig('sca.png', dpi=300, bbox_inches='tight')

fig, ax =plt.subplots(3,3,figsize=(12,10))
sns.scatterplot(y='Wyświetlenia', x='Taneczność', data=df_corr, ax=ax[0][0])
sns.scatterplot(y='Wyświetlenia', x='Energia', data=df_corr, ax=ax[0][1])
sns.scatterplot(y='Wyświetlenia', x='Głośność', data=df_corr, ax=ax[0][2])
sns.scatterplot(y='Wyświetlenia', x='Tekstowość', data=df_corr, ax=ax[1][0])
sns.scatterplot(y='Wyświetlenia', x='Akustyczność', data=df_corr, ax=ax[1][1])
sns.scatterplot(y='Wyświetlenia', x='Instrumentalność', data=df_corr, ax=ax[1][2])
sns.scatterplot(y='Wyświetlenia', x='Żywość', data=df_corr, ax=ax[2][0])
sns.scatterplot(y='Wyświetlenia', x='Pozytywność', data=df_corr, ax=ax[2][1])
sns.scatterplot(y='Wyświetlenia', x='Czas trwania', data=df_corr, ax=ax[2][2])
fig.savefig('scaYT.png', dpi=300, bbox_inches='tight')



