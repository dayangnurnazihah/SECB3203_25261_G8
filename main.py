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


# ------------------------------------------------------------
# MODEL EVALUATION 
# ------------------------------------------------------------

from sklearn.metrics import precision_score, recall_score, f1_score

print("\nLogistic Regression:")
log_accuracy = accuracy_score(y_test, log_test_pred)
log_precision = precision_score(y_test, log_test_pred)
log_recall = recall_score(y_test, log_test_pred)
log_f1 = f1_score(y_test, log_test_pred)
log_roc_auc = roc_auc_score(y_test, log_test_proba)

print(f"Accuracy: {log_accuracy:.4f}")
print(f"Precision: {log_precision:.4f}")
print(f"Recall: {log_recall:.4f}")
print(f"F1-Score: {log_f1:.4f}")
print(f"ROC-AUC: {log_roc_auc:.4f}")

print("\nRandom Forest:")
rf_accuracy = accuracy_score(y_test, rf_test_pred)
rf_precision = precision_score(y_test, rf_test_pred)
rf_recall = recall_score(y_test, rf_test_pred)
rf_f1 = f1_score(y_test, rf_test_pred)
rf_roc_auc = roc_auc_score(y_test, rf_test_proba)

print(f"Accuracy: {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall: {rf_recall:.4f}")
print(f"F1-Score: {rf_f1:.4f}")
print(f"ROC-AUC: {rf_roc_auc:.4f}")

# Compare models
print("\nMODEL COMPARISON:")
if rf_f1 > log_f1:
    print("Random Forest performs better (higher F1-Score)")
    best_model_name = "Random Forest"
else:
    print("Logistic Regression performs better (higher F1-Score)")
    best_model_name = "Logistic Regression"

# ------------------------------------------------------------
# OVER-FITTING, UNDER-FITTING DIAGNOSIS
# ------------------------------------------------------------

# Get training predictions
log_train_pred = log_reg_model.predict(X_train)
rf_train_pred = rf_model.predict(X_train)

# Calculate training accuracies
log_train_acc = accuracy_score(y_train, log_train_pred)
rf_train_acc = accuracy_score(y_train, rf_train_pred)

# Calculate gaps
log_gap = log_train_acc - log_accuracy
rf_gap = rf_train_acc - rf_accuracy

print("\nLogistic Regression:")
print(f"Training Accuracy: {log_train_acc:.4f}")
print(f"Test Accuracy: {log_accuracy:.4f}")
print(f"Gap: {log_gap:.4f}")
if log_gap > 0.08:
    print("DIAGNOSIS: Potential over-fitting")
elif log_gap < 0.01:
    print("DIAGNOSIS: Potential under-fitting")
else:
    print("DIAGNOSIS: Good generalization")

print("\nRandom Forest:")
print(f"Training Accuracy: {rf_train_acc:.4f}")
print(f"Test Accuracy: {rf_accuracy:.4f}")
print(f"Gap: {rf_gap:.4f}")
if rf_gap > 0.08:
    print("DIAGNOSIS: Potential over-fitting")
elif rf_gap < 0.01:
    print("DIAGNOSIS: Potential under-fitting")
else:
    print("DIAGNOSIS: Good generalization")

# ------------------------------------------------------------
# RIDGE REGRESSION (L2 Regularization)
# ------------------------------------------------------------

from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

# Create and train Ridge Regression model
ridge_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("scaler", StandardScaler()), 
    ("model", RidgeClassifier(alpha=1.0, random_state=42))
])

ridge_model.fit(X_train, y_train)
ridge_test_pred = ridge_model.predict(X_test)
ridge_train_pred = ridge_model.predict(X_train) 

ridge_accuracy = accuracy_score(y_test, ridge_test_pred)
ridge_precision = precision_score(y_test, ridge_test_pred)
ridge_recall = recall_score(y_test, ridge_test_pred)
ridge_f1 = f1_score(y_test, ridge_test_pred)

train_acc = accuracy_score(y_train, ridge_train_pred) 
test_acc = accuracy_score(y_test, ridge_test_pred)
gap = train_acc - test_acc

print("\nRidge Regression:")
print(f"Training Accuracy: {train_acc:.4f}") 
print(f"Test Accuracy:     {test_acc:.4f}")
print(f"Gap:              {gap:.4f}") 
print(f"Precision: {ridge_precision:.4f}")
print(f"Recall: {ridge_recall:.4f}")
print(f"F1-Score: {ridge_f1:.4f}")

print("\nNote: Ridge regression uses L2 regularization to reduce over-fitting")
print("      by penalizing large coefficients, which improves generalization.")

# ------------------------------------------------------------
# GRID SEARCH 
# ------------------------------------------------------------

try:
    from sklearn.model_selection import GridSearchCV
    
    print("\nPerforming Grid Search for Random Forest...")
    
    # Define a simpler parameter grid
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [5, 10],
        'model__min_samples_split': [2, 5]
    }
    
    # Create grid search
    grid_search = GridSearchCV(
        rf_model,
        param_grid,
        cv=3,  # Reduced from 5 to 3 for speed
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Extract best parameters
    best_params = grid_search.best_params_
    
    # ------------------------------------------------------------
    #region MODEL REFINEMENT
    # ------------------------------------------------------------
    
    from decimal import Decimal, ROUND_HALF_UP

    print("\nInitial Random Forest Model:")
    print(f"Accuracy: {rf_accuracy}")
    print(f"F1-Score: {rf_f1}")
    
    # Create refined model with best parameters
    refined_rf_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_estimators=best_params['model__n_estimators'],
            max_depth=best_params['model__max_depth'],
            min_samples_split=best_params['model__min_samples_split']
        ))
    ])
    
    refined_rf_model.fit(X_train, y_train)
    refined_pred = refined_rf_model.predict(X_test)
    
    refined_accuracy = accuracy_score(y_test, refined_pred)
    refined_f1 = f1_score(y_test, refined_pred)
    
    print("\nRefined Random Forest Model (with optimized parameters):")
    print(f"Accuracy: {refined_accuracy}")
    print(f"F1-Score: {refined_f1}")
    
    # Calculate improvement
    if rf_accuracy > 0:
        accuracy_improvement = ((refined_accuracy - rf_accuracy) / rf_accuracy) * 100
        print(f"\nAccuracy Improvement: {accuracy_improvement:+.2f}%")
    
    if rf_f1 > 0:
        f1_improvement = ((refined_f1 - rf_f1) / rf_f1) * 100
        print(f"F1-Score Improvement: {f1_improvement:+.2f}%")
    
        
except Exception as e:
    print(f"Grid Search failed with error: {e}")
    print("Using default parameters for refinement...")
    
    # Simple refinement without grid search
    print("\n5. MODEL REFINEMENT:")
    
    # Create a simpler model to reduce over-fitting
    simple_rf_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42
        ))
    ])
    
    simple_rf_model.fit(X_train, y_train)
    simple_pred = simple_rf_model.predict(X_test)
    simple_accuracy = accuracy_score(y_test, simple_pred)
    
    print(f"Initial Random Forest Accuracy: {rf_accuracy}")
    print(f"Simplified Random Forest Accuracy: {simple_accuracy}")
