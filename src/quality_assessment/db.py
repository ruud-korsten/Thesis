import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

db_config = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASS'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

def fetch_table(table_name: str) -> pd.DataFrame:
    try:
        with psycopg2.connect(**db_config) as conn:
            query = f"SELECT * FROM {table_name};"
            return pd.read_sql(query, conn)
    except Exception as e:
        print(f"Error fetching table: {e}")
        return None