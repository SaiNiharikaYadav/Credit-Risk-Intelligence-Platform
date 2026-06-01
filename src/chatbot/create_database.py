import sqlite3
import pandas as pd
import os

# Project root folder
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

# Database path
DB_PATH = os.path.join(
    BASE_DIR,
    "credit_risk.db"
)
print("BASE_DIR =", BASE_DIR)
print("DB_PATH =", DB_PATH)

# CSV paths
APP_PATH = os.path.join(
    BASE_DIR,
    "data",
    "application_train.csv"
)

BUREAU_PATH = os.path.join(
    BASE_DIR,
    "data",
    "bureau.csv"
)

PREVIOUS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "previous_application.csv"
)

print("Loading datasets...")

app = pd.read_csv(APP_PATH)
bureau = pd.read_csv(BUREAU_PATH)
previous = pd.read_csv(PREVIOUS_PATH)
print("Application rows:", len(app))
print("Bureau rows:", len(bureau))
print("Previous rows:", len(previous))

print("Connecting to database...")

conn = sqlite3.connect(DB_PATH)

app.to_sql(
    "application_train",
    conn,
    if_exists="replace",
    index=False
)

bureau.to_sql(
    "bureau",
    conn,
    if_exists="replace",
    index=False
)

previous.to_sql(
    "previous_application",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("Database created successfully!")