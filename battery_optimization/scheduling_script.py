from battery_optimization.data_sources import PostgresDataSource
from battery_optimization.optimization import (
    Battery,
    BatteryOptimization,
    System,
)


def optimization_scheduler(start_time, optimization_horizon, dispatch):
    db_url = "postgresql://admin:admin@127.0.0.1:5432/db"
    ds = PostgresDataSource(db_url)
    data = ds.get_forecast_data(start_time, optimization_horizon)
    battery = Battery(capacity=2600, max_charge=1000, max_discharge=1000)
    system = System(
        data["timestamp"].to_numpy(),
        data["solar"].to_numpy(),
        data["demand"].to_numpy(),
        data["price"].to_numpy(),
    )
    optimizer = BatteryOptimization(battery, system)
    initial_soc = ds.get_initial_soc(start_time)
    if initial_soc is None:
        initial_soc = 1300
    results_df = optimizer.solve(initial_soc=initial_soc, dispatch=dispatch)
    ds.update_optimization_results(results_df)

    return data, results_df


if __name__ == "__main__":
    # for start_time in pd.date_range("2023-01-01", "2023-01-31", freq="1D"):
    #     optimization_scheduler(start_time, 36)
    data, results = optimization_scheduler("2023-01-15", 36, dispatch=True)
    print(results.head(5))
