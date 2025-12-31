import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay
)

# ============================================================
#region LOAD DATASET
# ============================================================
dt = pd.read_csv("survey_lung_cancer.csv")

pd.set_option("display.max_columns", 20)

print("Dataset preview:")
print(dt.head())

print("\nDataset info:")
print(dt.info())


# ============================================================
#region DESCRIPTIVE STATISTICS 
# ============================================================
print("\nDescriptive statistics:")
print(dt.describe())

print("\nMissing values (True = missing):")
print(dt.isnull())

print("\nCount of missing values per column:")
print(dt.isnull().sum())


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
#region DATA CLEANING
# ============================================================

dt_clean = dt.copy()

dt_clean = dt_clean.fillna(dt_clean.mode().iloc[0])

print("\nMissing values after cleaning:")
print(dt_clean.isnull().sum())

yes_no_cols = [col for col in dt_clean.columns if dt_clean[col].dtype == "object"]

for col in yes_no_cols:
    dt_clean[col] = dt_clean[col].replace({
        "YES": 1, "NO": 0,
        "Yes": 1, "No": 0,
        "Y": 1, "N": 0
    })

dt_clean = dt_clean.apply(pd.to_numeric, errors="ignore")


num_cols = dt_clean.select_dtypes(include=[np.number]).columns.tolist()

TARGET_COL = "LUNG_CANCER"

if TARGET_COL in num_cols:
    num_cols.remove(TARGET_COL)


# ============================================================
# AGE GROUPING 
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
#region ANOVA TEST 
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
#region CORRELATION ANALYSIS 
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
#region PREPROCESSING PIPELINE 
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
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
#region LOGISTIC REGRESSION 
# ============================================================

log_reg_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

log_reg_model.fit(X_train, y_train)

log_test_pred = log_reg_model.predict(X_test)
log_test_proba = log_reg_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Performance:")
print("Accuracy:", accuracy_score(y_test, log_test_pred))
print("ROC-AUC:", roc_auc_score(y_test, log_test_proba))
print("\nClassification report:")
print(classification_report(y_test, log_test_pred))


# ============================================================
#region RANDOM FOREST CLASSIFIER 
# ============================================================
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42
    ))
])

rf_model.fit(X_train, y_train)

rf_test_pred = rf_model.predict(X_test)
rf_test_proba = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Performance:")
print("Accuracy:", accuracy_score(y_test, rf_test_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_test_proba))
print("\nClassification report:")
print(classification_report(y_test, rf_test_pred))


# ============================================================
#region MODEL COMPARISON VISUALIZATION
# ============================================================
models = ["Logistic Regression", "Random Forest"]
auc_scores = [
    roc_auc_score(y_test, log_test_proba),
    roc_auc_score(y_test, rf_test_proba)
]

plt.figure()
plt.bar(models, auc_scores)
plt.title("ROC-AUC Comparison Between Models")
plt.ylabel("ROC-AUC Score")
plt.ylim(0, 1)
plt.show()


# ============================================================
#region CONFUSION MATRIX 
# ============================================================
ConfusionMatrixDisplay.from_predictions(y_test, rf_test_pred)
plt.title("Random Forest Confusion Matrix")
plt.show()


# ============================================================
#region PREDICTION & DECISION MAKING
# ============================================================
sample = X_test.iloc[[0]]

risk_probability = rf_model.predict_proba(sample)[0][1]

print("\nPredicted probability of lung cancer:", round(risk_probability, 3))

if risk_probability >= 0.5:
    print("Decision: High risk of lung cancer")
else:
    print("Decision: Low risk of lung cancer")
