import pandas as pd

# Load dataset
df = pd.read_csv("data/application_train.csv")

print("="*50)
print("DATASET SHAPE")
print(df.shape)

print("\n" + "="*50)
print("FIRST 5 ROWS")
print(df.head())

print("\n" + "="*50)
print("TARGET DISTRIBUTION")
print(df["TARGET"].value_counts())

print("\n" + "="*50)
print("MISSING VALUES")
print(df.isnull().sum().sort_values(ascending=False).head(20))