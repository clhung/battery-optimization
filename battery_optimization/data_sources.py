import pandas as pd
from sqlalchemy import create_engine, text


class PostgresDataSource:
    """
    Load data from postgres data source

    Args:
        db_url:
    """

    def __init__(self, db_url):
        self.db_url = db_url
        self.engine = create_engine(db_url)

    def get_forecast_data(self, start_time, horizon):
        """
        Loads system data that are needed for scheduling

        Args:
            start_time:
            horizon: optimization horizon in hours
        """
        end_time = pd.to_datetime(start_time) + pd.Timedelta(hours=horizon)
        try:
            query = f"""
            SELECT * FROM forecast
            WHERE timestamp >= '{start_time}' AND timestamp < '{end_time}'
            """
            df = pd.read_sql(query, self.engine)

            if df.empty:
                raise ValueError("No data retrieved from the database.")

            return df

        except Exception as e:
            print(f"Error loading data from {self.db_url}: {e}")
            return None

    def update_optimization_results(self, results_df):
        tmp_table = "optimization_results_tmp"
        results_df.to_sql(
            tmp_table, self.engine, if_exists="replace", index=False
        )

        # merge data from tmp table
        upsert_query = text(
            f"""
            INSERT INTO optimization_results (timestamp, charge, discharge, soc, grid_import, solar_gen)
            SELECT timestamp, charge, discharge, soc, grid_import, solar_gen FROM {tmp_table}
            ON CONFLICT (timestamp)
            DO UPDATE SET
                charge = EXCLUDED.charge,
                discharge = EXCLUDED.discharge,
                soc = EXCLUDED.soc,
                grid_import = EXCLUDED.grid_import,
                solar_gen = EXCLUDED.solar_gen"""  # noqa
        )
        with self.engine.begin() as conn:
            conn.execute(upsert_query)

    def get_optimization_results(self, start_time, end_time):
        query = f"""
            SELECT * FROM optimization_results
            WHERE timestamp >= '{start_time}' AND timestamp < '{end_time}'
        """
        return pd.read_sql(query, self.engine)

    def get_initial_soc(self, timestamp):
        query = f"""
            SELECT soc FROM optimization_results
            WHERE timestamp = '{timestamp}'
        """
        df = pd.read_sql(query, self.engine)
        if df.empty:
            return None
        else:
            return df.loc[0, "soc"]
