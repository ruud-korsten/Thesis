import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from data_quality_tool.config.logging_config import get_logger

logger = get_logger()

load_dotenv()

def fetch_table(table_name: str, db_type: str = None) -> pd.DataFrame:
    db_name = db_type or os.getenv('DB_NAME')
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASS')
    host = os.getenv('DB_HOST', 'localhost')
    port = os.getenv('DB_PORT', '5432')

    db_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    engine = create_engine(db_url)

    query = f"SELECT * FROM {table_name};"
    logger.info("Fetching table '%s' from database '%s'", table_name, db_name)
    try:
        # Use chunking if the table is very large (optional)
        # chunks = []
        # for chunk in pd.read_sql(query, engine, chunksize=10000):
        #     chunks.append(chunk)
        # return pd.concat(chunks, ignore_index=True)

        df = pd.read_sql(query, engine)
        logger.info("Successfully fetched %d rows from table '%s'", len(df), table_name)
        return df
    except Exception as e:
        logger.exception("Error fetching table '%s': %s", table_name, str(e))
        return None
