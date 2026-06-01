import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

print("Loading dataset...")

df = pd.read_csv("data/application_train.csv")

print("Original Shape:", df.shape)

# Separate target
y = df["TARGET"]
X = df.drop("TARGET", axis=1)

# Store original feature names
original_features = X.columns.tolist()

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

print(f"Numeric Columns: {len(numeric_cols)}")
print(f"Categorical Columns: {len(categorical_cols)}")

# Fill missing values
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

# Save medians and modes for prediction UI
numeric_defaults = X[numeric_cols].median()

categorical_defaults = {}

for col in categorical_cols:
    categorical_defaults[col] = X[col].mode()[0]

# One-hot encoding
print("Encoding categorical features...")

X_encoded = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True
)

print("Encoded Shape:", X_encoded.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training XGBoost...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=11,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

pred_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, pred))

print(
    "ROC AUC:",
    roc_auc_score(y_test, pred_prob)
)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save everything

joblib.dump(model, "models/xgboost_model.pkl")

joblib.dump(
    original_features,
    "models/original_features.pkl"
)

joblib.dump(
    X_encoded.columns.tolist(),
    "models/model_columns.pkl"
)

joblib.dump(
    numeric_defaults,
    "models/numeric_defaults.pkl"
)

joblib.dump(
    categorical_defaults,
    "models/categorical_defaults.pkl"
)

feature_importance = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    "models/feature_importance.csv",
    index=False
)

print("\nSaved:")
print("xgboost_model.pkl")
print("original_features.pkl")
print("model_columns.pkl")
print("numeric_defaults.pkl")
print("categorical_defaults.pkl")
print("feature_importance.csv")