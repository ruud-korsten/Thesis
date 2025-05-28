import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from data_quality_tool.logging_config import get_logger

logger = get_logger()

load_dotenv()

# Load environment variables into a separate copy to avoid mutation
base_db_config = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASS'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}


def fetch_table(table_name: str, db_type: str = None) -> pd.DataFrame:
    logger.info("Fetching table '%s' from database (override dbname: %s)", table_name,
                db_type or base_db_config['dbname'])

    db_config = base_db_config.copy()
    if db_type:
        db_config['dbname'] = db_type

    try:
        with psycopg2.connect(**db_config) as conn:
            query = f"SELECT * FROM {table_name};"
            logger.debug("Executing query: %s", query)
            df = pd.read_sql(query, conn)
            logger.info("Successfully fetched %d rows from table '%s'", len(df), table_name)
            return df
    except Exception as e:
        logger.exception("Error fetching table '%s': %s", table_name, str(e))
        return None
