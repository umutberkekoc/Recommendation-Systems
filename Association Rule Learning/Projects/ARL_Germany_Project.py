import pandas as pd
pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)
#pd.set_option("display.max_rows", None)
pd.set_option("display.expand_frame_repr", False)
from mlxtend.frequent_patterns import apriori, association_rules
df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

def show_info(dataframe):
    print("*** HEAD ***")
    print(dataframe.head())
    print("*** SHAPE ***")
    print(dataframe.shape)
    print("*** INFO ***")
    print(dataframe.info())
    print("*** NA ***")
    print(dataframe.isnull().sum())
    print("*** DESCRIPTIVE STATS. ***")
    print(dataframe.describe().T)
    print("*** VALUE COUNTS """)
    print(dataframe.value_counts())

show_info(df)

# Data Preprocessing:

print(df[df["StockCode"] == "POST"])
df = df.drop(df[df["StockCode"] == "POST"].index)  # 1. way
df = df[~df["StockCode"].astype(str).str.contains("POST")]  # 2.way
df.info()

df.dropna(inplace=True)
print(df.isnull().sum())

df = df[~df["Invoice"].str.contains("C", na=False)]
print("C" in df["Invoice"])

df = df[df["Price"] > 0]
print(df.describe().T)

def outlier_thresholds(dataframe, variable):
    q1 = dataframe[variable].quantile(0.01)
    q3 = dataframe[variable].quantile(0.99)
    interquantile_range = q3 - q1
    up_limit = q3 + 1.5 * interquantile_range
    low_limit = q1 - 1.5 * interquantile_range
    return low_limit, up_limit

def change_with_th(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit

change_with_th(df, "Price")
change_with_th(df, "Quantity")
print(df.describe().T)


# ARL Data Structures

print(df["Country"].unique())
df_germany = df[df["Country"] == "Germany"]
print(df_germany)

print(df_germany.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}))
print(df_germany.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack())
print(df_germany.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().fillna(0))
print(df_germany.groupby(["Invoice", "Description"]).
      agg({"Quantity": "sum"}).unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0))

def description_or_stockcode(dataframe, is_stock=False):
    if is_stock:
        return (dataframe.groupby(["Invoice", "StockCode"]).agg({"Quantity": "sum"}).
                unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0))
    else:
        return (dataframe.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).
                unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0))




df_germany_matrix = description_or_stockcode(df_germany, is_stock=True)

print(df_germany_matrix.head())
df_germany_matrix = df_germany_matrix.astype(bool)
df_germany_matrix = df_germany_matrix.droplevel(0, axis=1)
print(df_germany_matrix.shape)


x = apriori(df_germany_matrix, min_support=0.01, use_colnames=True)
print(x)

rules = association_rules(x, min_threshold=0.01, metric="support")
print(rules)
print(rules.shape)

# functionalization to creating rules

def create_rules(dataframe):
    print(dataframe["Country"].unique())
    country = str(input("Enter Country Name"))
    min_support = float(input("Enter The Minimum Support Level"))
    min_th = float(input("Enter The Minimum Threshold Level"))
    dataframe = dataframe[dataframe["Country"] == country]
    dataframe = description_or_stockcode(df)
    dataframe = dataframe.astype(bool)
    dataframe = dataframe.droplevel(0, axis=1)
    x = apriori(dataframe, min_support=min_support, use_colnames=True)
    rules = association_rules(x, min_threshold=min_th, metric="support")
    return rules

rules = create_rules(df)

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code]["Description"].values[0]
    print("Product Name: {}".format(product_name))
check_id(df, stock_code=71053)



product_id = 21086
sorted_rules = rules.sort_values("lift", ascending=False)
print(sorted_rules.head(20))
recommendation_list = []
for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])


print(recommendation_list[0])

def arl_recommender(rule, product_id, sort_by="lift", rec_count=1):
    sorted_rules = rule.sort_values(sort_by, ascending=False)
    recommendation_list=[]
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_count]


arl_recommender(rules, product_id=21086)
arl_recommender(rules, product_id=22747)
arl_recommender(rules, product_id=23245, sort_by="confidence", rec_count=2)
check_id(df, 22745)


check_id(df, stock_code=arl_recommender(rules, product_id=21086)[0])
check_id(df, stock_code=arl_recommender(rules, product_id=22747)[0])
check_id(df, stock_code=arl_recommender(rules, product_id=23245)[0])
