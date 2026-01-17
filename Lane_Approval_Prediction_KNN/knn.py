################################################
# KNN
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


################################################
# 1. Exploratory Data Analysis
################################################

df = pd.read_csv("loan_data.csv")

df.head()
df.shape
df.describe().T
df["loan_status"].value_counts()

################################################
# 2. Data Preprocessing & Feature Engineering
################################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

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
    return cat_cols, num_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

def analys_for_cats(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({col: dataframe.groupby(col)[target].mean()}), end="\n\n\n")

analys_for_cats(df,"loan_status",cat_cols)

def analys_for_num(dataframe, target, num_cols):
    for col in num_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({col: dataframe.groupby(target)[col].mean()}), end="\n\n\n")

analys_for_num(df,"loan_status",num_cols)


# Aykırı Gözlem Analizi
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for i in num_cols:
    print(check_outlier(df, i))

def get_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    outliers = dataframe.loc[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit), col_name]
    return outliers

for i in num_cols:
    print(get_outliers(df, i))

# Eksik Değer Analizi
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns
missing_values_table(df)

# Rare Encoder Analizi
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "loan_status",cat_cols)

cat_cols = [x for x in cat_cols if x!="loan_status"]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first,dtype=int)
    return dataframe
df = one_hot_encoder(df, cat_cols)

df.head()

df.shape

# Numerik değişkenler için standartlaştırma
scalar = StandardScaler()
df[num_cols] = scalar.fit_transform(df[num_cols])


################################################
# 3. Modeling & Prediction
################################################

y = df["loan_status"]
X = df.drop("loan_status", axis=1)


knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)

################################################
# 4. Model Evaluation
################################################

#Confusion matrix için y_pred
y_pred = knn_model.predict(X)

# AUC için y_prob, roc curveun tek bir sayısal değer ile ifade edilişidir.
y_proba = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
# acc 0.93
# f1 0.83
#AUC
rec_auc_score = roc_auc_score(y, y_proba)
# 0.9763765799999999

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=['accuracy','f1','roc_auc'])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#0.8941
#0.748
#0.9185

knn_model.get_params()

################################################
# 5. Hyperparameter Optimization
################################################

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {'n_neighbors': range(2,50)}

# her bir parametre için deneyecek ve hatalara bakacaz. n_jobs-1 ile işlemci en yüksek hızda kullan, verbose 1 demek rapor ver demek
knn_gs_best = GridSearchCV(knn_model, knn_params, cv=5, n_jobs=-1, verbose= 1).fit(X, y)

knn_gs_best.best_params_
#17

################################################
# 6. Final Model
################################################
#iki yıldızla kolayca atama işlemini yapıyoruz, tek tek yapmak çok parametre olduğunda çok zor oluyor.

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final, X, y, cv=5, scoring=['accuracy','f1','roc_auc'])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#0.9007
#0.7524
#0.9469
#başarıyı artırmış olduk.

random_user = X.sample(1)

knn_final.predict(random_user)