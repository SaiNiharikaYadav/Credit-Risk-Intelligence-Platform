import sqlite3
import pandas as pd
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

DB_PATH = os.path.join(
    BASE_DIR,
    "credit_risk.db"
)

def run_query(sql):

    print("Using database:")
    print(DB_PATH)

    if not sql:
        raise Exception("No SQL generated")

    if sql.startswith("-- ERROR"):
        raise Exception(sql)

    conn = sqlite3.connect(DB_PATH)

    try:
        result = pd.read_sql_query(sql, conn)
        return result
    finally:
        conn.close()