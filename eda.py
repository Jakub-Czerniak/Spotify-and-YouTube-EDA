import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import log10
import numpy as np
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv('Spotify_Youtube.csv', index_col=0)

# drop unnecessary columns
df = df.drop(['Url_spotify', 'Url_youtube', 'Description', 'Channel'], axis=1)
df_start = df.copy()
df_start.rename(
    columns={'Danceability': 'Taneczność', 'Energy': 'Energia', 'Loudness': 'Głośność', 'Speechiness': 'Tekstowość',
             'Acousticness': 'Akustyczność', 'Instrumentalness': 'Instrumentalność', 'Liveness': 'Żywość',
             'Valence': 'Pozytywność', 'Duration_ms': 'Czas trwania', 'Stream': 'Odsłuchania', 'Views': 'Wyświetlenia',
             'Likes': 'Polubienia', 'Comments': 'Komentarze', 'Licensed': 'Licencjonowany',
             'official_video': 'Oficjalny film'}, inplace=True)

# missing values
df = df.dropna()

'''# replace string with int
df['Licensed'] = df['Licensed'].astype(int)
df['official_video'] = df['official_video'].astype(int)
df['Album_type'] = df['Album_type'].replace('album', 0)
df['Album_type'] = df['Album_type'].replace('single', 1)
df['Album_type'] = df['Album_type'].replace('compilation', 2)
# drop duplicates
df = df.drop_duplicates(subset='Uri', keep='first')'''

# correlation
'''df_corr = df.copy()
df_corr = df_corr.drop(['Artist', 'Track', 'Album', 'Album_type', 'Title', 'Key', 'Uri'], axis=1)
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
'''

# conversion to ordinal data
df_ordinal = df.copy()
df_ordinal = df_ordinal.drop(['Title', 'Uri', 'Album', 'Track', 'Artist'], axis=1)
df_ordinal.rename(columns={'Danceability': 'Taneczność', 'Energy': 'Energia', 'Loudness': 'Głośność',
                           'Speechiness': 'Tekstowość', 'Acousticness': 'Akustyczność',
                           'Instrumentalness': 'Instrumentalność', 'Liveness': 'Żywość',
                           'Valence': 'Pozytywność', 'Duration_ms': 'Czas trwania',
                           'Stream': 'Odsłuchania', 'Views': 'Wyświetlenia', 'Likes': 'Polubienia',
                           'Comments': 'Komentarze', 'Licensed': 'Licencjonowany',
                           'official_video': 'Oficjalny film', 'Key': 'Klucz', 'Album_type': 'Typ albumu'},
                  inplace=True)


def range_to_1_5(x, ranges):
    if ranges[4][0] <= x <= ranges[4][1]:
        return '5'
    if ranges[3][0] <= x < ranges[3][1]:
        return '4'
    if ranges[2][0] <= x < ranges[2][1]:
        return '3'
    if ranges[1][0] <= x < ranges[1][1]:
        return '2'
    if ranges[0][0] <= x < ranges[0][1]:
        return '1'


pd.options.display.float_format = '{:.2f}'.format # uncomment to print ranges without scientific notation
np.set_printoptions(suppress=True)


def convert_numeric_to_ordinal(df, column_name):
    # converts numeric data to ordinal 1-5 data
    column_max = df[column_name].max()
    column_min = df[column_name].min()
    column_range = column_max - column_min
    ranges = np.empty((5, 2))
    ranges[4] = tuple((column_min + column_range * 4 / 5, column_max))
    ranges[3] = tuple((column_min + column_range * 3 / 5, column_min + column_range * 4 / 5))
    ranges[2] = tuple((column_min + column_range * 2 / 5, column_min + column_range * 3 / 5))
    ranges[1] = tuple((column_min + column_range * 1 / 5, column_min + column_range * 2 / 5))
    ranges[0] = tuple((column_min, column_min + column_range * 1 / 5))
    print('Ranges of ' + column_name + ':')
    print(ranges)
    df[column_name] = df[column_name].apply(lambda x: range_to_1_5(x, ranges))
    return df


df_ordinal = convert_numeric_to_ordinal(df_ordinal, 'Taneczność')
df_ordinal = convert_numeric_to_ordinal(df_ordinal, 'Energia')
df_ordinal = convert_numeric_to_ordinal(df_ordinal, 'Głośność')
df_ordinal = convert_numeric_to_ordinal(df_ordinal, 'Akustyczność')
df_ordinal = convert_numeric_to_ordinal(df_ordinal, 'Pozytywność')
df_ordinal = convert_numeric_to_ordinal(df_ordinal, 'Tempo')

df_ordinal['Tekstowość'] = df_ordinal['Tekstowość'].mask(df_ordinal['Tekstowość'] < 0.13, 0)
df_ordinal['Tekstowość'] = df_ordinal['Tekstowość'].mask(df_ordinal['Tekstowość'] > 0.98, 3)
df_ordinal['Tekstowość'] = df_ordinal['Tekstowość'].mask(df_ordinal['Tekstowość'].between(0.13, 0.66), 1)
df_ordinal['Tekstowość'] = df_ordinal['Tekstowość'].mask(df_ordinal['Tekstowość'].between(0.66, 0.98), 2)
df_ordinal['Tekstowość'] = df_ordinal['Tekstowość'].astype(int).astype(str)
df_ordinal['Tekstowość'] = df_ordinal['Tekstowość'].astype('category')

df_ordinal['Instrumentalność'] = df_ordinal['Instrumentalność'].mask(df_ordinal['Instrumentalność'] <= 0.5, 0)
df_ordinal['Instrumentalność'] = df_ordinal['Instrumentalność'].mask(df_ordinal['Instrumentalność'] > 0.5, 1)
df_ordinal['Instrumentalność'] = df_ordinal['Instrumentalność'].astype(int).astype(str)
df_ordinal['Instrumentalność'] = df_ordinal['Instrumentalność'].astype('category')

df_ordinal['Żywość'] = df_ordinal['Żywość'].mask(df_ordinal['Żywość'] <= 0.8, 0)
df_ordinal['Żywość'] = df_ordinal['Żywość'].mask(df_ordinal['Żywość'] > 0.8, 1)
df_ordinal['Żywość'] = df_ordinal['Żywość'].astype(int).astype(str)
df_ordinal['Żywość'] = df_ordinal['Żywość'].astype('category')

df_ordinal['Klucz'] = df_ordinal['Klucz'].astype(int).astype(str)
df_ordinal['Klucz'] = df_ordinal['Klucz'].astype('category')

df_ordinal['Czas trwania'] = df_ordinal['Czas trwania'] = df_ordinal['Czas trwania'].apply(
    lambda x: str(int(x // (1000 * 60))))
df_ordinal['Czas trwania'] = df_ordinal['Czas trwania'].astype('category')


def convert_numeric_to_ordinal_log(df, column_name):
    # converts numeric values to log10 rounded downwards
    df = df.loc[(df[column_name] != 0)]
    df[column_name] = df[column_name].apply(lambda x: str(int(log10(x) // 1)))
    return df


df_ordinal = convert_numeric_to_ordinal_log(df_ordinal, 'Odsłuchania')
df_ordinal = convert_numeric_to_ordinal_log(df_ordinal, 'Wyświetlenia')
df_ordinal = convert_numeric_to_ordinal_log(df_ordinal, 'Polubienia')
df_ordinal = convert_numeric_to_ordinal_log(df_ordinal, 'Komentarze')

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_ordinal.head())

df_dummies = pd.get_dummies(df_ordinal,
                            columns=['Typ albumu', 'Taneczność', 'Energia', 'Klucz', 'Głośność', 'Tekstowość',
                                     'Akustyczność', 'Instrumentalność', 'Żywość', 'Pozytywność', 'Tempo',
                                     'Czas trwania', 'Wyświetlenia', 'Polubienia', 'Komentarze', 'Odsłuchania'])
df_dummies = df_dummies.drop(['Żywość_0', 'Instrumentalność_0', 'Tekstowość_0'], axis=1)
fp_results = fpgrowth(df_dummies, min_support=0.4, use_colnames=True, max_len=None, verbose=0)
res_df = association_rules(fp_results, min_threshold=0.8)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(res_df)
res_df.to_csv('asos(0.8).txt')
fp_results.to_csv('support(0.4).txt')


df_ordinal_spotify_top = df_ordinal.loc[df_ordinal['Odsłuchania'] == '9']
df_ordinal_spotify_top = df_ordinal_spotify_top.drop(['Odsłuchania', 'Polubienia', 'Komentarze', 'Oficjalny film', 'Licencjonowany'], axis=1)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_ordinal_spotify_top.head())
df_dummies_spotify = pd.get_dummies(df_ordinal_spotify_top,
                            columns=['Typ albumu', 'Taneczność', 'Energia', 'Klucz', 'Głośność', 'Tekstowość',
                                     'Akustyczność', 'Instrumentalność', 'Żywość', 'Pozytywność', 'Tempo',
                                     'Czas trwania', 'Wyświetlenia'])
df_dummies_spotify = df_dummies_spotify.drop(['Żywość_0', 'Instrumentalność_0', 'Tekstowość_0'],axis=1)
fp_results = fpgrowth(df_dummies_spotify, min_support=0.4, use_colnames=True, max_len=None, verbose=0)
res_df = association_rules(fp_results, min_threshold=0.8)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(res_df)
res_df.to_csv('asos_spotify(0.8).txt')
fp_results.to_csv('spotify_top_support(0.4).txt')


df_ordinal_yt_top = df_ordinal.loc[df_ordinal['Wyświetlenia'] == '9']
df_ordinal_yt_top = df_ordinal_yt_top.drop(['Wyświetlenia', 'Oficjalny film', 'Licencjonowany'], axis=1)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_ordinal_yt_top.head())
df_dummies_yt = pd.get_dummies(df_ordinal_yt_top,
                            columns=['Typ albumu', 'Taneczność', 'Energia', 'Klucz', 'Głośność', 'Tekstowość',
                                     'Akustyczność', 'Instrumentalność', 'Żywość', 'Pozytywność', 'Tempo',
                                     'Czas trwania', 'Odsłuchania', 'Polubienia', 'Komentarze'])
df_dummies_yt = df_dummies_yt.drop(['Żywość_0', 'Instrumentalność_0', 'Tekstowość_0'],axis=1)
fp_results = fpgrowth(df_dummies_yt, min_support=0.4, use_colnames=True, max_len=None, verbose=0)
res_df = association_rules(fp_results, min_threshold=0.8)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(res_df)
res_df.to_csv('asos_yt(0.8).txt')
fp_results.to_csv('yt_top_support(0.4).txt')
