import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

print("Loading model...")

model = joblib.load("models/xgboost_model.pkl")

df = pd.read_csv("data/application_train.csv")

df = df.select_dtypes(include=["number"])
df = df.fillna(df.median())

X = df.drop("TARGET", axis=1)

print("Generating SHAP values...")

explainer = shap.TreeExplainer(model)

sample = X.sample(1000, random_state=42)

shap_values = explainer.shap_values(sample)

shap.summary_plot(
    shap_values,
    sample,
    show=False
)

plt.savefig("documents/shap_summary.png")

print("SHAP summary saved!")