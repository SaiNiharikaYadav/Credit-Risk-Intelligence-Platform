from sql_generator import generate_sql
from query_runner import run_query

question = "How many customers defaulted?"

sql = generate_sql(question)

print("\nSQL:")
print(sql)

result = run_query(sql)

print("\nResult:")
print(result)