import pandas as pd
pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)
#pd.set_option("display.max_rows", None)
pd.set_option("display.expand_frame_repr", False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
print(df.head())
print(df.describe().T)
print(df.isnull().sum())

# Association Rule Learning:
# 1- Data preprocessing
# 2- Creating ARL data structures
# 3- Association Rule Analysis
# 4- preparing Script
# 5- product Recommendation application


# 1- Data Preprocessing:

def show_info(dataframe):
    print("***** HEAD *****")
    print(dataframe.head())
    print("***** SHAPE *****")
    print(dataframe.shape)
    print("***** INFO *****")
    print(dataframe.info())
    print("***** NA *****")
    print(dataframe.isnull().sum())
    print("***** DESCRIPTIVE STATS. *****")
    print(dataframe.describe().T)

show_info(df)

def outlier_thresholds(dataframe, variable):
    q1 = dataframe[variable].quantile(0.01)  # %1'lik dilimdekileri baskılama
    q3 = dataframe[variable].quantile(0.99)  # %99'luk dilimdekileri baskılama
    range = q3 - q1
    upper_limit = q3 + 1.5 * range
    lower_limit = q1 - 1.5 * range
    return lower_limit, upper_limit

def change_with_out_th(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > upper_limit), variable] = upper_limit
    dataframe.loc[(dataframe[variable] < lower_limit), variable] = lower_limit


def data_preprocessing(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    change_with_out_th(dataframe, "Quantity")
    change_with_out_th(dataframe, "Price")
    return dataframe

df = data_preprocessing(df)
show_info(df)



# 2- Creating ARL Data Structures:

df_fr = df[df["Country"] == "France"]
show_info(df_fr)

print(df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}))

print(df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack())

print(df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().fillna(0))

print(df_fr.groupby(["Invoice", "Description"]).
      agg({"Quantity": "sum"}).unstack().fillna(0).
      applymap(lambda x: 1 if x > 0 else 0))

def desc_or_stockcode(dataframe, stockcode=True):
    if stockcode:
        return (dataframe.groupby(["Invoice", "StockCode"]).agg({"Quantity": "sum"}).
                unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0))
    else:
        return (dataframe.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).
                unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0))

df_fr_matrix = desc_or_stockcode(df_fr)

# Check StockCode:

def check_stockcode(dataframe, stock_code):
    product = dataframe[dataframe["StockCode"] == stock_code]["Description"].values[0]
    print("Product:", product)

check_stockcode(df_fr, 22728)


# 3- Association Rule Analysis:
df_fr_matrix = df_fr_matrix.astype(bool)
df_fr_matrix = df_fr_matrix.droplevel(0, axis=1)

x = apriori(df_fr_matrix, min_support=0.01, use_colnames=True)
print(x.head())

rules = association_rules(x, min_threshold=0.01, metric="support")
print(rules)

print(rules.sort_values("confidence", ascending=False))

print(rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.1) &
            (rules["lift"] > 5)].sort_values("confidence", ascending=False).head(10))


# 4- preparing Script
df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

def show_info(dataframe):
    print("***** HEAD *****")
    print(dataframe.head())
    print("***** SHAPE *****")
    print(dataframe.shape)
    print("***** INFO *****")
    print(dataframe.info())
    print("***** NA *****")
    print(dataframe.isnull().sum())
    print("***** DESCRIPTIVE STATS. *****")
    print(dataframe.describe().T)

def outlier_thresholds(dataframe, variable):
    q1 = dataframe[variable].quantile(0.01)  # %1'lik dilimdekileri baskılama
    q3 = dataframe[variable].quantile(0.99)  # %99'luk dilimdekileri baskılama
    range = q3 - q1
    upper_limit = q3 + 1.5 * range
    lower_limit = q1 - 1.5 * range
    return lower_limit, upper_limit

def change_with_out_th(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > upper_limit), variable] = upper_limit
    dataframe.loc[(dataframe[variable] < lower_limit), variable] = lower_limit


def data_preprocessing(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    change_with_out_th(dataframe, "Quantity")
    change_with_out_th(dataframe, "Price")
    return dataframe

def desc_or_stockcode(dataframe, stockcode=True):
    if stockcode:
        return (dataframe.groupby(["Invoice", "StockCode"]).agg({"Quantity": "sum"}).
                unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0))
    else:
        return (dataframe.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).
                unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0))


def check_stockcode(dataframe, stock_code):
    product = dataframe[dataframe["StockCode"] == stock_code]["Description"].values[0]
    print("Product:", product)

def create_rules(dataframe, stockcode=True, country="France"):
    dataframe = dataframe[dataframe["Country"] == country]
    dataframe = desc_or_stockcode(dataframe)
    dataframe = dataframe.astype(bool)
    dataframe = dataframe.droplevel(0, axis=1)
    x = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(x, min_threshold=0.01, metric="support")
    return rules


show_info(df)
df = data_preprocessing(df)

rules = create_rules(df, stockcode=True)

print(rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.1) & (rules["lift"] > 5)].
      sort_values("confidence", ascending=False).head(10))


# 5- product Recommendation application:

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_count]

check_stockcode(df_fr, 21094)
arl_recommender(rules, product_id=21094, rec_count=1)
arl_recommender(rules, product_id=21094, rec_count=2)
arl_recommender(rules, product_id=21094, rec_count=3)