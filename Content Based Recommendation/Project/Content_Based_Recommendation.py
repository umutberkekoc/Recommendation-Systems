# Content Based Recommendation Filtering Project:

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
pd.set_option("display.width", 600)
pd.set_option("display.max_columns", None)
df = pd.read_csv("movies_metadata.csv", low_memory=False)
Data = df.astype(bool)



# Data Understanding:
def show_info(dataframe):
    print("*** HEAD ***")
    print(dataframe.head())
    print("*** SHAPE ***")
    print(dataframe.shape)
    print("*** SIZE ***")
    print(dataframe.size)
    print("*** NA ***")
    print(dataframe.isnull().sum())
    print("*** #OF Unique ***")
    print(dataframe.nunique())
    print("*** DESCRIPTIVE STATS. ***")
    print(dataframe.describe().T)
    print("*** COLUMNS ***")
    print(dataframe.columns)

show_info(df)

# Data Preprocessing:

df.isnull().sum()
df.shape
df["overview"].nunique()
df["overview"].isnull().sum()

df["title"].nunique()
df["title"].value_counts()

df["overview"].fillna("", inplace=True)
df["overview"].isnull().sum()

# Creating TF-IDF Matrix:
tf_idf = TfidfVectorizer(stop_words="english")      # creating tf_idf
tf_idf_matrix = tf_idf.fit_transform(df["overview"])  # creating tf_idf_matrix

print(tf_idf.get_feature_names_out())  # return names of tf_idf
tf_idf_matrix = tf_idf_matrix.astype(np.float32)  # writed to avoiding ram error
print(tf_idf_matrix.toarray())
print(tf_idf_matrix.shape)  # check the shape of tf_idf_matrix

# Creating Cosine Similarity:
cos_sim = cosine_similarity(tf_idf_matrix, tf_idf_matrix)  # creating cosine similarity
print(cos_sim)
print(cos_sim.shape)  # check the shape of cos_sim
print(cos_sim[1])

# Recommendation According to Similarities:
df.index
indices = pd.Series(df.index, index=df["title"])

    # checking duplicates
print(indices.index.value_counts())  # there are duplicates
print(indices.index.duplicated().any())
print(indices[indices.index == "Cinderella"])


indices = indices[~indices.index.duplicated(keep="last")]
print(indices.index.value_counts())

movie_name = "Sherlock Holmes"
movie_index = indices["Sherlock Holmes"]

print(cos_sim[movie_index])
print(cos_sim[indices["Sherlock Holmes"]])  # 2.way

# Creating similaritiy scores dataframe
similarity_scores = pd.DataFrame(cos_sim[movie_index])
similarity_scores.columns = ["score"]

best_rec_index = similarity_scores.sort_values("score", ascending=False)[1:11].index


best_rec_name = df["title"].iloc[best_rec_index]
best_rec_name=best_rec_name.reset_index()
best_rec_name.columns = ["film_index", "film_name"]
best_rec_name["corr_score"] = similarity_scores["score"].sort_values(ascending=False)[1:11].values
print(best_rec_name)


# Preparing Script:

def show_info(dataframe):
    print("*** HEAD ***")
    print(dataframe.head())
    print("*** SHAPE ***")
    print(dataframe.shape)
    print("*** SIZE ***")
    print(dataframe.size)
    print("*** NA ***")
    print(dataframe.isnull().sum())
    print("*** #OF Unique ***")
    print(dataframe.nunique())
    print("*** DESCRIPTIVE STATS. ***")
    print(dataframe.describe().T)
    print("*** COLUMNS ***")
    print(dataframe.columns)

show_info(df)

def data_preprocessing(dataframe):
    if dataframe["overview"].isnull().sum().any() == True:
        return dataframe["overview"].fillna("")

def content_based_recommender(title, dataframe, language="english"):
    df_= pd.read_csv("movies_metadata.csv", low_memory=False)
    df = df_.copy()
    Data = df.astype(bool)
    dataframe = data_preprocessing(dataframe)

    tf_idf = TfidfVectorizer(stop_words=language)
    tf_idf_matrix = tf_idf.fit_transform(dataframe["overview"])
    tf_idf_matrix = tf_idf_matrix.astype(np.float32)
    print(tf_idf.get_feature_names_out())
    print(tf_idf_matrix.toarray())

    cos_sim = cosine_similarity(tf_idf_matrix, tf_idf_matrix)

    indices = pd.Series(dataframe.index, index=dataframe["title"])
    if indices.index.duplicated().any() == True:
        indices = indices[~indices.duplicated(keep="last")]

    movie_index = indices[title]

    similarity_scores = pd.DataFrame(cos_sim[movie_index])
    similarity_scores.columns = ["score"]

    num = int(input("enter a number for recommend film"))
    best_rec_index = similarity_scores.sort_values("score", ascending=False)[1:num].index

    best_rec_name = df["title"].iloc[best_rec_index]
    best_rec_name = best_rec_name.reset_index()
    best_rec_name.columns = ["film_index", "film_name"]
    best_rec_name["corr_score"] = similarity_scores["score"].sort_values(ascending=False)[1:num].values

    return best_rec_name

content_based_recommender("Sherlock Holmes", df)