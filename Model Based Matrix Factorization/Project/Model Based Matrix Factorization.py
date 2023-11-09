import pandas as pd
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 700)

movie = pd.read_csv("movie.csv")
rating = pd.read_csv("rating.csv")
df = movie.merge(rating, on="movieId", how="left")

print(df.head())

# ****** Data Pre-Processing ******

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df["movieId"].isin(movie_ids)]

print("Sample Data:\n", sample_df)

user_movie_df = sample_df.pivot_table(index="userId", columns="title", values="rating")
print("User Movie DF:\n", user_movie_df)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(sample_df[["userId", "movieId", "rating"]], reader)

# ***** Modelling *****
train_set, test_set = train_test_split(data, test_size=0.25)
svd_model = SVD()
svd_model.fit(train_set)
prediction = svd_model.test(test_set)
print("Prediction:\n", prediction)

print("RMSE:", accuracy.rmse(prediction))

svd_model.predict(uid=1.0, iid=541, verbose=True)
svd_model.predict(uid=1.0, iid=356, verbose=True)
print(sample_df[sample_df["userId"] == 1])

# ***** Model Tuning *****
param_grid = {"n_epochs": [5, 10, 20],           # default is 20
              "lr_all": [0.002, 0.005, 0.007]}   # default is 0.007


gs = GridSearchCV(SVD,
        param_grid,
        measures=["rmse", "mae"],  # root mean squared error, mean absoulte error
        cv=3,  # cross valdiation. (take 3 piece from data, use 2 for train and use 1 for test then change it)
        n_jobs=1,  # use all processor at full performance
        joblib_verbose=True)  # report

gs.fit(data)
print("Best Grid Search CV Score:\t", gs.best_score["rmse"])
print("Best Grid Search CV Parameters:\t", gs.best_params["rmse"])

# ***** Final Model and Prediction *****
dir(svd_model)
print("n_epochs:", svd_model.n_epochs)

svd_model_final = SVD(**gs.best_params["rmse"])
data = data.build_full_trainset()
svd_model_final.fit(data)

print(sample_df[sample_df["userId"] == 1])
svd_model_final.predict(uid=1, iid=541, verbose=True)

print(sample_df[sample_df["userId"] == 2])
svd_model_final.predict(uid=2, iid=541, verbose=True)

print(sample_df[sample_df["userId"] == 7])
svd_model_final.predict(uid=7, iid=356, verbose=True)