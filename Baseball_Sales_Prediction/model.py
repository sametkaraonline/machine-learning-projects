"""
Baseball Salary Prediction Study - Abdülsamet Kara
"""

# 1. Main Anaylises
# 2. Outlier Anaylises
# 3. Missing Values Anaylises
# 4. Encoding
# 5. Corelation Anaylises
# 6. Normal Scalar
# 7. Model

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x : "%.3f" % x)
pd.set_option("display.width",500)

############################################
# Main Anaylises
############################################
df = pd.read_csv("hitters.csv")

df.shape
df.head()

#There are missing values in tagret variable and non-numeratic values are in other columns.
# catch categoric and numeric cols

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Returns the names of categorical, numeric and categorical but cardinal variables in the data set.
    Note Categorical variables include categorical variables with numeric appearance.

    Parameters
    ------
        dataframe: dataframe
                Variable names of the dataframe to be taken
        cat_th: int, optional
                class threshold for numeric but categorical variables
        car_th: int, optinal
                class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                List of cardinal variables with categorical appearance

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of the 3 return lists equals the total number of variables: cat_cols + num_cols + cat_but_car = number of variables
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# target analysis for numeric cols
def analys(dataframe, target, num_cols):
    for col in num_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({col: dataframe.groupby(target)[col].mean()}), end="\n\n\n")

analys(df,"Salary",num_cols)

# target analysis for categoric cols
def analys_for_cats(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({col: dataframe.groupby(col)[target].mean()}), end="\n\n\n")

analys_for_cats(df,"Salary",cat_cols)

############################################
# Outlier Anaylises
############################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    intequantiles_range  = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * intequantiles_range
    low_limit = quartile1 - 1.5 * intequantiles_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def remove_outlier(dataframe, colname):
    low_limit, up_limit = outlier_thresholds(dataframe, colname)
    df_without_ouliers = dataframe[~((dataframe[colname] < low_limit) | (dataframe[colname] > up_limit))]
    return df_without_ouliers

def replace_outliers_with_thresholds(dataframe, colname):
    # outlier_thresholds fonksiyonu ile sınırları alıyoruz
    low_limit, up_limit = outlier_thresholds(dataframe, colname)

    # Kolondaki aykırı değerleri alt ve üst sınırlarla değiştiriyoruz
    dataframe[colname] = dataframe[colname].apply(
        lambda x: low_limit if x < low_limit else (up_limit if x > up_limit else x)
    )

    return dataframe

def get_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    outliers = dataframe.loc[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit), col_name]
    return outliers

for col in num_cols:
    print(col, check_outlier(df,col))

for col in num_cols:
    print(col, get_outliers(df,col))

for col in num_cols:
    new_df = replace_outliers_with_thresholds(df, col)

#If you wish, you can delete 11 columns, but i didnt
df.shape
new_df.shape

df = new_df

############################################
# Missing Values Anaylises
############################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)

# Fill in missing variables by League and Division (A,E)(A,W)(N,E)(N,W)

df.loc[(df["Salary"].isnull()) & (df["League"] == "A") & (df["Division"] == "E"), "Salary"] = df.groupby(["League","Division"])["Salary"].mean()["A","E"]

df.loc[(df["Salary"].isnull()) & (df["League"] == "A") & (df["Division"] == "W"), "Salary"] = df.groupby(["League","Division"])["Salary"].mean()["A","W"]

df.loc[(df["Salary"].isnull()) & (df["League"] == "N") & (df["Division"] == "E"), "Salary"] = df.groupby(["League","Division"])["Salary"].mean()["N","E"]

df.loc[(df["Salary"].isnull()) & (df["League"] == "N") & (df["Division"] == "W"), "Salary"] = df.groupby(["League","Division"])["Salary"].mean()["N","W"]

df.shape

############################################
# Encoding
############################################

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first,dtype=int)
    return dataframe

df = one_hot_encoder(df, cat_cols)

df.dtypes


############################################
# Feature Extraction
############################################

df["Avg_AtBat_per_year"] = df["AtBat"] / df["Years"]

df["Avg_Hits_per_year"] = df["Hits"] / df["Years"]

df["Avg_HmRun_per_year"] = df["HmRun"] / df["Years"]

df["Avg_RBI_per_year"] = df["RBI"] / df["Years"]

df["Avg_Walks_per_year"] = df["Walks"] / df["Years"]

df["Career_Hit_Rate"] = df["CHits"] / df["CAtBat"]

df["Career_HmRun_Rate"] = df["CHmRun"] / df["CAtBat"]

df["Career_RBI_Rate"] = df["CRBI"] / df["CAtBat"]

df["Career_Walk_Rate"] = df["CWalks"] / df["CAtBat"]

df["Batting_efficiency"] = df["Hits"] / df["AtBat"]

df["Homerun_efficiency"] = df["HmRun"] / df["AtBat"]

df["RBI_efficiency"] = df["RBI"] / df["AtBat"]

df["Walks_efficiency"] = df["Walks"] / df["AtBat"]

############################################
# Corelation Anaylises
############################################
# We need to look after comvert to categoric columns to numeric cols.

corr = df.corr()
df.corr()["Salary"].sort_values(ascending=False)

############################################
# Normal Scalar
############################################

scalar = StandardScaler()
df[num_cols] = scalar.fit_transform(df[num_cols])

df.head()

############################################
# Model
############################################
y = df["Salary"]
X = df.drop(["Salary"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

reg_model = LinearRegression().fit(X_train, y_train)

#(b - bias)
float(reg_model.intercept_)

#(w - weights)
reg_model.coef_[0]

# Predictions on training data
y_pred_train = reg_model.predict(X_train)

# RMSE (Root Mean Squared Error) on Train Set
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

# R^2 Score on Train Set
train_r2 = reg_model.score(X_train, y_train)

# Predictions on test data
y_pred_test = reg_model.predict(X_test)

# RMSE (Root Mean Squared Error) on Test Set
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

# R^2 Score on Test Set
test_r2 = reg_model.score(X_test, y_test)

# Print the results with formatted output
print("############# Model Performance #############")
print(f"Train R² Score: {train_r2:.4f}")
print(f"Test R² Score: {test_r2:.4f}")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print("##############################################")