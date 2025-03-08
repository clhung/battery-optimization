import pandas as pd

from battery_optimization.data_sources import PostgresDataSource
from battery_optimization.optimization import (
    Battery,
    BatteryOptimization,
    System,
)


def optimization_scheduler(start_time, optimization_horizon):
    db_url = "postgresql://admin:admin@127.0.0.1:5432/db"
    ds = PostgresDataSource(db_url)
    data = ds.get_forecast_data(start_time, optimization_horizon)
    battery = Battery(capacity=250, max_charge=50, max_discharge=50)
    system = System(
        data["timestamp"].to_numpy(),
        data["solar"].to_numpy(),
        data["demand"].to_numpy(),
        data["price"].to_numpy(),
    )
    optimizer = BatteryOptimization(battery, system)
    initial_soc = ds.get_initial_soc(start_time)
    if initial_soc is None:
        initial_soc = 50
    results_df = optimizer.solve(initial_soc=initial_soc)
    ds.update_optimization_results(results_df)


if __name__ == "__main__":
    for start_time in pd.date_range("2023-01-01", "2023-01-31", freq="1D"):
        optimization_scheduler(start_time, 36)
