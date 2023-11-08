# Armut Association Rule Learning Project

import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 700)
from mlxtend.frequent_patterns import apriori, association_rules

# Adım 1: Data Preprocessing:
df_ = pd.read_csv("armut_data.csv")
df = df_.copy()

def show_info(dataframe):
    print("*** HEAD ***")
    print(dataframe.head())
    print("*** SHAPE ***")
    print(dataframe.shape)
    print("*** SIZE ***")
    print(dataframe.size)
    print("*** INFO ***")
    print(dataframe.info())
    print("*** NA ***")
    print(dataframe.isnull().sum())
    print("*** DESCRIPTIVE STATISTICS ***")
    print(dataframe.describe().T)

show_info(df)

df["hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)
print(df.head())


df["date"] = df["CreateDate"].astype("datetime64[ns]").dt.to_period("M")
df["ID"] = df["UserId"].astype(str) + "_" + df["date"].astype(str)
print(df.head())


df.nunique()
df.shape

def create_df_matrix(dataframe):
    agg = str(input("Enter one of variable from the dataframe"))
    return (dataframe.groupby(["ID", "hizmet"]).agg({agg: "count"}).
                 unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0))

df_matrix = create_df_matrix(df)
print(df_matrix.iloc[0:5, 0:10])


def create_rules(dataframe, use_col=True):
    min_support = float(input("Enter the Minimum Support (0.01)"))
    min_th = float(input("Enter the Minimum Threshold (0.01)"))
    dataframe = dataframe.astype(bool)
    # dataframe = dataframe.droplevel(0, axis=1)
    x = apriori(dataframe, min_support=min_support, use_colnames=use_col)
    rules = association_rules(x, min_threshold=min_th, metric="support")
    return rules

rules  = create_rules(df_matrix)
print(rules.head())

print(rules.sort_values("confidence", ascending=False).head(20))
print(rules[(rules["support"] > 0.01) & (rules["confidence"] > 0.10) & (rules["lift"] > 4)])

#Adım 3: Creating ARL RECOMMENDER FUNCTION

def arl_recommender(df_rule, sort_by="lift", rec_count=1):
    hizmet = str(input("Enter the Servide ID")) + "_" + str(input("Enter the Category ID"))
    print("Alınan Hizmet:", hizmet)
    recommendation_list = []
    sorted_rule = df_rule.sort_values(sort_by, ascending=False)
    for index, product in enumerate(sorted_rule["antecedents"]):
        for i in list(product):
            if i == hizmet:
                recommendation_list.append(list(sorted_rule.iloc[index]["consequents"])[0])
                
    return "Önerilen Hzimet:", recommendation_list[0:rec_count]

arl_recommender(rules)

print(df[(df["ServiceId"] == "2") & (df["Category_Id"] == 0)].sort_values("CreateDate", ascending=False).head(1))

# Printing User Information by Sevice and Category ID:

def person_info(dataframe, asc=False, order_by="CreateDate", head=5):
    service_id = int(input("Enter the Service ID"))
    category_id = int(input("Enter the Category ID"))
    return (dataframe[(dataframe["ServiceId"] == service_id) & (dataframe["CategoryId"] == category_id)].
            sort_values(order_by, ascending=asc)).head(head)

person_info(df)
person_info(df, head=20)

def get_user_id(dataframe):
    service_id = int(input("Enter the Service ID"))
    category_id = int(input("Enter the Category ID"))

    user_id = pd.DataFrame()
    user_id = dataframe[(dataframe["ServiceId"] == service_id) & (dataframe["CategoryId"] == category_id)]["UserId"].drop_duplicates()
    user_id.columns = ["user_id"]
    return user_id

user_id = get_user_id(df)

print(get_user_id(df).to_csv("user_id.csv"))
print(user_id.to_csv("user_id2.csv"))  # 2. way