import pandas as pd
from sqlalchemy import create_engine

from battery_optimization.optimization import (
    Battery,
    BatteryOptimization,
    System,
)


class PostgresDataLoader:
    """Load system data from a posgres connection"""

    def __init__(self, db_url):
        """
        Initializes the PostgresDataLoader with database connection

        Args:
             db_url: posgres url
        """
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


if __name__ == "__main__":
    db_url = "postgresql://admin:admin@127.0.0.1:5432/db"
    loader = PostgresDataLoader(db_url)

    data = loader.load_system_data()

    battery = Battery(capacity=50, max_charge=10, max_discharge=10)
    system = System(
        data["solar"].to_numpy(),
        data["demand"].to_numpy(),
        data["price"].to_numpy(),
    )

    optimizer = BatteryOptimization(battery, system)
    results = optimizer.solve(initial_soc=25)
