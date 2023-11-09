import pandas as pd
import random
pd.set_option("display.width", 600)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
movie = pd.read_csv("movie.csv")
rating = pd.read_csv("rating.csv")

# 1- Data Pre-Processing

print(movie.head())
print(rating.head())
df = movie.merge(rating, on="movieId", how="left")

df["title"].isnull().sum()

print(df.shape)
print(df["title"].nunique())
print(df["title"].value_counts())

def create_user_movie_df(dataframe, th=5000):
    evaluated_movie_count = pd.DataFrame(dataframe["title"].value_counts())
    rare_movie_names = evaluated_movie_count[evaluated_movie_count["count"] <= th].index
    common_movie_df = dataframe[~dataframe["title"].isin(rare_movie_names)]
    print("DF Shape: {}\tRare Movie Name Length: {}\tCommon Movie DF: {}".
          format(dataframe.shape[0], len(rare_movie_names), common_movie_df.shape[0]))
    print(common_movie_df)

    user_movie_df = common_movie_df.pivot_table(index="userId", columns="title", values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df(df)


user_ids_list = []
user_ids_list = user_movie_df.index
random_user_id = random.choice(user_ids_list)
print("Random User ID:", random_user_id)

# OR
random_user_id = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)  # 2. way
print("Random User ID:", random_user_id)


# Creating random user dataframe
random_user_df = user_movie_df[user_movie_df.index == random_user_id]


# 2- Determining the watched films by random user
# 1. way
random_user_watched_films = random_user_df.columns[random_user_df.notna().any()].tolist()
print("Watched Films by Random User:", random_user_watched_films)
print("Number of Watched Films by Random User", len(random_user_watched_films))

# 2.way
random_user_watched_films = []
for index, i in enumerate(random_user_df.iloc[0, :]):
    if i > 0:
        film_name = random_user_df.columns[index]
        random_user_watched_films.append(film_name)

print("Watched Films by Random User:", random_user_watched_films)
print("Number of Watched Films by Random User", len(random_user_watched_films))


# Checking Watched Films and Rating By random user:
# 1. way
print(user_movie_df.loc[user_movie_df.index == random_user_id,
                                                 user_movie_df.columns == "Aladdin (1992)"])

# 2. way
def check_film_random_user(movie_name):
    if movie_name in random_user_watched_films:
        for index, i in enumerate(random_user_df):
            if i == movie_name:
                value = random_user_df.iloc[0, index]
                print("{} Movie Named {} Watched By UserID and his/her Rate is: {}".format(film_name, random_user_df.index[0], value))
    else:
        print("{} Adlı Film {} UserID'li Kişi Tarafından İzlenmedi".format(film_name, random_user_df.index[0]))


check_film_random_user("Avatar (2009)")

# Creating only watched films by random user dataframe
movies_watched_df = user_movie_df[random_user_watched_films]
print(movies_watched_df.shape, len(movies_watched_df))

# findin other users who watched at least 60 % common movie
user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]

print(user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False))

print(user_movie_count[user_movie_count["movie_count"] == len(random_user_watched_films)].count())


same_users = user_movie_count[user_movie_count["movie_count"] > len(random_user_watched_films) * 80 / 100]["userId"]
print("Same Users:", same_users.shape)


final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(same_users)],
                      random_user_df[random_user_watched_films]])

final_df.index.nunique()
final_df.drop_duplicates(inplace=True)

corr_df = final_df.T.corr().unstack().sort_values()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ["user_id_1", "user_id_2"]
corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == random_user_id) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users_rating = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")

top_users_rating = top_users_rating[top_users_rating["userId"] != random_user_id]


# Weighted Average Recommendation Score
top_users_rating["weighted_rating"] = top_users_rating["corr"] * top_users_rating["rating"]

print(top_users_rating.groupby("movieId").agg({"weighted_rating": "mean"}))

recommendation_df = top_users_rating.groupby("movieId").agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

print(recommendation_df[recommendation_df["weighted_rating"] > 3.5])
movies_tobe_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)
movies_tobe_recommend.merge(movie[["movieId", "title"]])