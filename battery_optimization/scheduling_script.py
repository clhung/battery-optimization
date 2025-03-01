# from stable_baselines3 import PPO

from battery_optimization.data_sources import PostgresDataLoader
from battery_optimization.optimization import (
    Battery,
    BatteryOptimization,
    System,
)

# from dev.rl_optimization import SchedulerEnv

if __name__ == "__main__":
    db_url = "postgresql://admin:admin@127.0.0.1:5432/db"
    loader = PostgresDataLoader(db_url)

    data = loader.load_system_data()

    battery = Battery(capacity=50, max_charge=10, max_discharge=10)
    system = System(
        data["solar"].iloc[0:].to_numpy(),
        data["demand"].iloc[0:].to_numpy(),
        data["price"].iloc[0:].to_numpy(),
    )

    optimizer = BatteryOptimization(battery, system)
    results = optimizer.solve(initial_soc=25)
    results.to_csv("results/battery_optimization.csv", index=False)

    # Create Gym environment
    # env = SchedulerEnv(battery, system)
    #
    # # Train RL model using PPO
    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=100000)
    #
    # # Save trained model
    # model.save("models/ppo_battery_optimizer")
