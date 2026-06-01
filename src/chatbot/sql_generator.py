import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get Gemini API Key
api_key = os.getenv("GEMINI_API_KEY")

# Debug check
print("API Key Found:", api_key is not None)

# Configure Gemini
genai.configure(api_key=api_key)

# Create model
model = genai.GenerativeModel("gemini-2.5-flash")

# Database Schema
schema = """
application_train(
    SK_ID_CURR,
    TARGET,
    AMT_INCOME_TOTAL,
    AMT_CREDIT,
    AMT_ANNUITY
)

bureau(
    SK_ID_BUREAU,
    SK_ID_CURR,
    CREDIT_ACTIVE,
    CREDIT_DAY_OVERDUE
)

previous_application(
    SK_ID_PREV,
    SK_ID_CURR,
    AMT_APPLICATION,
    AMT_CREDIT
)
"""

def generate_sql(question):

    print("Inside generate_sql()")

    prompt = f"""
You are a SQL expert.

Database Schema:

{schema}

Convert the user's question into a valid SQLite SQL query.

Rules:
1. Return ONLY SQL.
2. Do not explain.
3. Use table names exactly as given.
4. Always generate complete SQL.

Question:
{question}
"""

    print("Sending request to Gemini...")

    try:

        response = model.generate_content(prompt)

        print("Received response.")

        sql = response.text.strip()

        # Remove markdown if Gemini returns ```sql
        sql = sql.replace("```sql", "")
        sql = sql.replace("```", "")
        sql = sql.strip()

        print("Generated SQL:")
        print(sql)

        return sql

    except Exception as e:

        print("ERROR:", str(e))

        return f"-- ERROR: {str(e)}"