import pandas as pd
from sqlalchemy import create_engine


class PostgresDataLoader:
    """
    Load data from postgres data source

    Args:
        db_url:
    """

    def __init__(self, db_url):
        self.db_url = db_url
        self.engine = create_engine(db_url)

    def load_system_data(self):
        """
        Loads system data that are needed for scheduling
        """
        try:
            query = "SELECT * FROM example_schedule"
            df = pd.read_sql(query, self.engine)

            if df.empty:
                raise ValueError("No data retrieved from the database.")

            return df

        except Exception as e:
            print(f"Error loading data from {self.db_url}: {e}")
            return None
