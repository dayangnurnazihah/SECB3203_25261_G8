# ============================================================
# LUNG CANCER PREDICTION
# Project Progress 3 & 4
# Exploratory Data Analysis + Model Development
# ============================================================

# -----------------------------
# IMPORT LIBRARIES
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# LOAD DATASET
# ============================================================
dt = pd.read_csv("survey_lung_cancer.csv")

pd.set_option("display.max_columns", 20)

print("Dataset preview:")
print(dt.head())

print("\nDataset info:")
print(dt.info())


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================
print("\nDescriptive statistics:")
print(dt.describe())


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================
print("\nMissing values (True = missing):")
print(dt.isnull())

print("\nCount of missing values per column:")
print(dt.isnull().sum())


# ============================================================
# BASIC VISUALIZATION
# ============================================================

# Gender distribution
plt.figure()
dt["GENDER"].value_counts().plot(
    kind="bar",
    color=["skyblue", "lightpink"]
)
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Lung cancer distribution
counts = dt["LUNG_CANCER"].value_counts()

plt.figure()
plt.pie(
    counts,
    labels=counts.index,
    autopct="%1.1f%%",
    colors=["lightcoral", "lightskyblue"],
    startangle=90,
    explode=[0.1, 0]
)
plt.title("Respondents Affected by Lung Cancer")
plt.axis("equal")
plt.show()


# ============================================================
# DATA CLEANING
# ============================================================

dt_clean = dt.copy()

# Fill missing values using mode (suitable for survey data)
dt_clean = dt_clean.fillna(dt_clean.mode().iloc[0])

print("\nMissing values after cleaning:")
print(dt_clean.isnull().sum())


# ============================================================
# FORMAT DATA (YES / NO → 1 / 0)
# ============================================================

yes_no_cols = [col for col in dt_clean.columns if dt_clean[col].dtype == "object"]

for col in yes_no_cols:
    dt_clean[col] = dt_clean[col].replace({
        "YES": 1, "NO": 0,
        "Yes": 1, "No": 0,
        "Y": 1, "N": 0
    })

# Convert all possible columns to numeric
dt_clean = dt_clean.apply(pd.to_numeric, errors="ignore")


# ============================================================
# NUMERICAL FEATURE IDENTIFICATION
# ============================================================
num_cols = dt_clean.select_dtypes(include=[np.number]).columns.tolist()

TARGET_COL = "LUNG_CANCER"

if TARGET_COL in num_cols:
    num_cols.remove(TARGET_COL)


# ============================================================
# AGE GROUPING (FEATURE ENGINEERING)
# ============================================================
dt_clean["AGE_GROUP"] = pd.cut(
    dt_clean["AGE"],
    bins=[0, 30, 50, 100],
    labels=["Young", "Adult", "Senior"]
)

print("\nAge group distribution:")
print(dt_clean["AGE_GROUP"].value_counts())



# ------------------------------------------------------------
# GROUPING BY TARGET
# ------------------------------------------------------------
grouped_stats = dt_clean.groupby(TARGET_COL)[num_cols].agg(["mean", "median", "std"])
print("\nGrouped statistics by lung cancer outcome:")
print(grouped_stats)


# ------------------------------------------------------------
# ANOVA TEST
# ------------------------------------------------------------
anova_results = {}

for col in num_cols:
    group0 = dt_clean[dt_clean[TARGET_COL] == 0][col]
    group1 = dt_clean[dt_clean[TARGET_COL] == 1][col]

    if len(group0) > 1 and len(group1) > 1:
        f_stat, p_value = stats.f_oneway(group0, group1)
        anova_results[col] = {
            "F-statistic": f_stat,
            "p-value": p_value
        }

anova_df = pd.DataFrame(anova_results).T
print("\nANOVA Results:")
print(anova_df)


# ------------------------------------------------------------
# CORRELATION ANALYSIS
# ------------------------------------------------------------
corr_matrix = dt_clean[num_cols + [TARGET_COL]].corr()

plt.figure(figsize=(10, 6))
plt.title("Correlation Matrix")
plt.imshow(corr_matrix, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# FEATURE–TARGET SPLIT
# ------------------------------------------------------------
X = dt_clean.drop(columns=[TARGET_COL])
y = dt_clean[TARGET_COL]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()


# ------------------------------------------------------------
# PREPROCESSING PIPELINE 
# ------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)


# ------------------------------------------------------------
# TRAIN–TEST SPLIT
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# SIMPLE & MULTIPLE LINEAR REGRESSION
# ============================================================

linear_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)

y_train_pred = linear_model.predict(X_train)
y_test_pred = linear_model.predict(X_test)

print("\nLinear Regression Performance:")
print("Train R²:", r2_score(y_train, y_train_pred))
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Test R²:", r2_score(y_test, y_test_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))


# Visualization
plt.figure()
plt.scatter(y_train, y_train_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Actual vs Predicted (Training Set)")
plt.show()


# ============================================================
# POLYNOMIAL REGRESSION WITH PIPELINE
# ============================================================
poly_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", LinearRegression())
])

poly_model.fit(X_train, y_train)

y_poly_pred = poly_model.predict(X_test)

print("\nPolynomial Regression Performance:")
print("R²:", r2_score(y_test, y_poly_pred))
print("MSE:", mean_squared_error(y_test, y_poly_pred))


# ============================================================
# MODEL COMPARISON VISUALIZATION
# ============================================================
models = ["Linear Regression", "Polynomial Regression"]
r2_scores = [
    r2_score(y_test, y_test_pred),
    r2_score(y_test, y_poly_pred)
]

mse_scores = [
    mean_squared_error(y_test, y_test_pred),
    mean_squared_error(y_test, y_poly_pred)
]

plt.figure()
plt.bar(models, r2_scores)
plt.title("R² Comparison Between Models")
plt.ylabel("R² Score")
plt.show()

plt.figure()
plt.bar(models, mse_scores)
plt.title("MSE Comparison Between Models")
plt.ylabel("Mean Squared Error")
plt.show()


# ============================================================
# PREDICTION & DECISION MAKING
# ============================================================
sample = X_test.iloc[[0]]
prediction = poly_model.predict(sample)[0]

print("\nSample Prediction Value:", prediction)

if prediction >= 0.5:
    print("Decision: High risk of lung cancer")
else:
    print("Decision: Low risk of lung cancer")
