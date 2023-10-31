import pandas as pd
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 20)

# Data Preperation and Preprocessing
movie = pd.read_csv("movie.csv")
rating = pd.read_csv("rating.csv")
df = movie.merge(rating, on="movieId", how="left")

df.isnull().sum()
df.dropna(inplace=True)
Data = df.astype(bool)


print(df.head())
print(df.isnull().sum().any())
print(df.describe().T)


# Creating user_movie_df

print(df["title"].value_counts())

comment_counts = pd.DataFrame(df["title"].value_counts())
print("Comment Counts:", comment_counts)

rare_movies = comment_counts[comment_counts["title"] <= 10000].index
print("Rare Movies:", rare_movies)

common_movies = df[~df["title"].isin(rare_movies)]

print("df Shape: {}\nRare Movie Shape: {}\nCommon Movie Shape: {}".format(df.shape, rare_movies.shape, common_movies.shape))


user_movie_df = common_movies.pivot_table("rating", "userId", "title")  # value, row, column
user_movie_df = common_movies.pivot_table(index="userId", columns="title", values="rating")
print("User Movie DF:", user_movie_df)


movie_name = "Matrix, The (1999)"
print(user_movie_df.corrwith(user_movie_df[movie_name]).sort_values(ascending=False).head(10))


movie_name = "Mission: Impossible (1996)"
movie_name = user_movie_df[movie_name]
print(user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10))


# Functionalization of The Process:
def create_user_movie_df():
    min_count = int(input("Enter The Minimum Comment Level!"))
    import pandas as pd
    df1 = pd.read_csv("movie.csv")
    df2 = pd.read_csv("rating.csv")
    on = None
    how = str(input("Enter The Join Type (left)"))
    for i in df1.columns:
        for j in df2.columns:
            if i == j:
                on = i
    df = df1.merge(df2, on=on, how=how)
    comment_count = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_count[comment_count["title"] <= min_count].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index="userId", columns="title", values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

def check_film_name(dataframe, film_name):
    return [i for i in dataframe.columns if film_name in i]

check_film_name(user_movie_df, film_name="Matrix")

def item_based_recommender(moivename, user_movie_df):
    movie_name = user_movie_df[moivename]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)", user_movie_df)




