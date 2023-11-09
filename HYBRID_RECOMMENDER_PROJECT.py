#############################################
# Project: Hybrid Recommender System
#############################################
import pandas as pd
import random
import numpy as np
pd.set_option("display.width", 600)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)

movie = pd.read_csv("movie.csv")
rating = pd.read_csv("rating.csv")

print(movie.head())
print(rating.head())



df = rating.merge(movie, on="movieId", how="left")
print(df.head())


print(df["title"].value_counts())  #
print(df.groupby("title").agg({"userId": "count"}).sort_values("userId", ascending=False))  # 2.way
movie_evaluation_num = pd.DataFrame(df["title"].value_counts())


rare_movies = movie_evaluation_num[movie_evaluation_num["count"] <= 4000].index
print("Rare Movie Names: {}\nNumber of Rare Movie: {}".format(rare_movies, len(rare_movies)))


frequent_movies = df[~df["title"].isin(rare_movies)]
print("Number of Frequent Movie:", frequent_movies["title"].nunique())

user_movie_df = frequent_movies.pivot_table(index="userId", columns="title", values="rating")
print("User Movie DF:", user_movie_df)



def create_usermovie_df(dataframe):
    th = int(input("Enter the Min Evaluation Value "))
    movie_evaluation_num = pd.DataFrame(dataframe["title"].value_counts())
    rare_movies = movie_evaluation_num[movie_evaluation_num["count"] <= th].index
    frequent_movies = dataframe[~dataframe["title"].isin(rare_movies)]
    user_movie_df = frequent_movies.pivot_table(index="userId", columns="title", values="rating")
    return user_movie_df

user_movie_df = create_usermovie_df(df)


#############################################


user_id_list = user_movie_df.index
random_user_id = random.choice(user_id_list)
print("Random User ID:", random_user_id)


random_user_df = user_movie_df[user_movie_df.index == random_user_id]
print("Random User DF (with all frequent movies):\n", random_user_df)


movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
print("Watched Movies: {}\nNumber of Watched Movies: {}".format(movies_watched, len(movies_watched)))

movies_watched2 = []
for index, i in enumerate(random_user_df.iloc[0, :]):
    if i > 0:
        movie_name = random_user_df.columns[index]
        movies_watched2.append(movie_name)
print("Movies Watched2:\n", movies_watched2)  # 2.way

#############################################


movies_watched_df = user_movie_df[movies_watched]
print("Movies Watched DF (all users with only watched films dataframe)\n", movies_watched_df)
print(movies_watched_df.columns.nunique())


user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
print("User Movie Count:\n", user_movie_count)

percentage = len(movies_watched) * 60 / 100

print(user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False))

users_same_movies = user_movie_count[user_movie_count["movie_count"] >= percentage]["userId"]
print("Users Same Movies:\n", users_same_movies)

#############################################

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])
final_df.index.nunique()
final_df = final_df.drop_duplicates()


corr_df = final_df.T.corr().unstack().sort_values()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ["user_id_1", "user_id_2"]
corr_df = corr_df.reset_index()

corr_df[corr_df["user_id_1"] == random_user_id]

top_users = (corr_df[(corr_df["user_id_1"] == random_user_id) & (corr_df["corr"] >= 0.55)][["user_id_2", "corr"]].
             reset_index(drop=True).sort_values("corr", ascending=False))

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)


top_users_rating = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")

top_users_rating = top_users_rating[top_users_rating["userId"] != random_user_id]

#############################################

top_users_rating["weighted_rating"] = top_users_rating["corr"] * top_users_rating["rating"]

print(top_users_rating.groupby("movieId").agg({"weighted_rating": "mean"}))

recommendation_df = top_users_rating.groupby("movieId").agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

print(recommendation_df[recommendation_df["weighted_rating"] > 2.5])
movies_tobe_recommend = recommendation_df[recommendation_df["weighted_rating"] > 2.5].sort_values("weighted_rating", ascending=False)
movies_tobe_recommend.merge(movie[["movieId", "title"]])


print(movies_tobe_recommend.merge(movie[["movieId", "title"]]).head())

#############################################
user = 108170

import pandas as pd
import datetime as dt
pd.set_option("display.width", 600)
pd.set_option("display.max_columns", None)
movie = pd.read_csv("movie.csv")
rating = pd.read_csv("rating.csv")
df = movie.merge(rating, on="movieId", how="left")
df.head()
df.info()

movie_id = (rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].
            sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0])


print(movie[movie["movieId"] == movie_id]["title"].values[0])
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
movie_df

user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)





