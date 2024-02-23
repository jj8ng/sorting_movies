import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn import preprocessing
from fuzzywuzzy import process

def data_removal(movie_df):
    movie_df.drop([19730, 29502, 35585], inplace=True)
    return movie_df

def data_norm(r):
    return 1 if r >= 1 else 0

def apri(input_movies, movies_df, ratings_df):
    apriori_result = []
    movies_df.dropna(subset=['title'], inplace=True)
    df = pd.merge(ratings_df, movies_df[['id', 'title']], left_on='movieId', right_on='id').drop(['timestamp', 'id'], axis=1)
    df = df.drop_duplicates(['userId', 'title'])
    df_pivot = df.pivot(index='userId', columns='title', values='rating').fillna(0).astype('int64').applymap(apriori_encoding)
    frequent_items = apriori(df_pivot, min_support=0.07, use_colnames=True)
    association_indicator = association_rules(frequent_items, metric="lift", min_threshold=1).sort_values(by=['lift'], ascending=False)
    
    for selected_movie in input_movies:
        df_selected = association_indicator[(association_indicator['antecedents'].apply(lambda x: len(x) == 1 and next(iter(x)) == selected_movie)) & (association_indicator['lift'] > 1.2)]
        recommended_movies = df_selected['consequents'].apply(lambda x: list(x)[0]).values
        apriori_result.extend(recommended_movies[:10])
    
    return apriori_result

def kmea(apriori_result, input_movies, movies_df):
    clusters = []
    kmeans_result = []
    numeric_df = movies_df[['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count', 'title']].dropna()
    df_numeric = numeric_df[numeric_df['vote_count'] > 25]
    df_numeric_scaled = preprocessing.MinMaxScaler().fit_transform(df_numeric.drop('title', axis=1))
    
    kmeans = KMeans(n_clusters=5).fit(df_numeric_scaled)
    df_numeric['cluster'] = kmeans.labels_
    
    for movie1 in input_movies:
        cluster_candid = df_numeric.loc[df_numeric["title"] == movie1, 'cluster'].values[0]
        clusters.append(cluster_candid)
    
    for movie2 in apriori_result:
        cluster_tmp = df_numeric.loc[df_numeric["title"] == movie2, 'cluster'].values[0]
        if cluster_tmp in clusters:
            kmeans_result.append(movie2)
    
    return kmeans_result

def main(input_movies):
    final_result = ""
    final_result += f"Selected movies ({len(input_movies)} movies) : " + ",".join(input_movies) + "\n\n"
    
    movies_df = pd.read_csv('data/movies_metadata.csv')
    ratings_df = pd.read_csv('data/ratings_small.csv')
    movies_df = drop_trash_data(movies_df)
    
    apriori_result = do_apriori(input_movies, movies_df, ratings_df)
    kmeans_result = do_kmeans(apriori_result, input_movies, movies_df)
    
    final_result += "A-priori & K-means clustering recommend movie : " + ",".join(kmeans_result) + "\n\n"
    
    with open("result.txt", "w") as f:
        f.write(final_result)
    
    return final_result

if __name__ == '__main__':
    input_movies = input("Enter the movie names (comma-separated): ").split(',')
    main(input_movies)