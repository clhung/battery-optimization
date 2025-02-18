import cvxpy as cp
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass


@dataclass
class Battery:
    capacity: float  # kWh
    max_charge: float  # kW
    max_discharge: float  # kW
    charge_efficiency: float = 0.9
    discharge_efficiency: float = 0.9


@dataclass
class System:
    solar: np.ndarray
    load: np.ndarray
    price: np.ndarray

    @property
    def timesteps(self):
        return len(self.load)


class BatteryOptimization:

    def __init__(self, battery: Battery, system: System):
        self.battery = battery
        self.system = system
        self.ts = system.timesteps

    def define_variables(self):
        self.charge = cp.Variable(self.ts, nonneg=True)
        self.discharge = cp.Variable(self.ts, nonneg=True)
        self.soc = cp.Variable(self.ts + 1, nonneg=True)
        self.grid_import = cp.Variable(self.ts, nonneg=True)
        self.z = cp.Variable(self.ts, boolean=True) # control charge and discharge

    def define_constraints(self, initial_soc: float):
        self.constraints = []

        # battery states
        self.constraints += [ self.soc[0] == initial_soc + self.charge[0] * self.battery.charge_efficiency - self.discharge[0] / self.battery.discharge_efficiency]
        self.constraints += [ self.soc <= self.battery.capacity ]

        for t in range(self.ts):
            self.constraints += [ self.soc[t + 1] == self.soc[t] + self.charge[t] * self.battery.charge_efficiency - self.discharge[t] / self.battery.discharge_efficiency]
            self.constraints += [self.charge <= self.z[t] * self.battery.max_charge]
            self.constraints += [self.discharge <= (1 - self.z[t]) * self.battery.max_discharge]

            # power balance
            self.constraints += [ self.grid_import[t] + self.system.solar[t] + self.discharge[t] == self.system.load[t] + self.charge[t] ]


    def solve(self, initial_soc):
        self.define_variables()
        self.define_constraints(initial_soc=initial_soc)
        obj = cp.Minimize(cp.sum(self.grid_import * self.system.price))
        problem = cp.Problem(obj, self.constraints)
        problem.solve(solver=cp.SCIPY)

        return {
            "charge_schedule": self.charge.value,
            "discharge_schedule": self.discharge.value,
            "soc_schedule": self.soc.value,
            "grid_import_schedule": self.grid_import.value
        }

    def plot_results(self, results):
        """Plots the optimization results using Plotly."""
        time = np.arange(self.ts)

        fig = go.Figure()

        # Battery SOC
        fig.add_trace(
            go.Scatter(x=time, y=results["soc_schedule"][:-1], mode='lines+markers', name="Battery SOC (kWh)"))

        # Charging and discharging power
        fig.add_trace(
            go.Scatter(x=time, y=results["charge_schedule"], mode='lines+markers', name="Charging Power (kW)"))
        fig.add_trace(
            go.Scatter(x=time, y=results["discharge_schedule"], mode='lines+markers', name="Discharging Power (kW)"))

        # Grid Import
        fig.add_trace(
            go.Scatter(x=time, y=results["grid_import_schedule"], mode='lines+markers', name="Grid Import (kW)"))

        fig.update_layout(title="Battery Optimization Results",
                          xaxis_title="Time (hours)",
                          yaxis_title="Power (kW)",
                          legend_title="Legend",
                          template="plotly_dark")

        fig.show()


# Example usage
if __name__ == "__main__":
    # Define battery parameters
    battery = Battery(capacity=50, max_charge=10, max_discharge=10)

    # Define energy system parameters (example input)
    solar_generation = np.array([0, 0, 1, 3, 7, 10, 12, 14, 12, 10, 8, 5, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # kW
    load_demand = np.array([5, 4, 6, 7, 10, 15, 18, 20, 22, 18, 15, 12, 10, 8, 7, 6, 6, 5, 4, 4, 4, 4, 4, 5])  # kW
    price_signal = np.array([0.10, 0.10, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.30, 0.28,
                             0.25, 0.22, 0.20, 0.18, 0.15, 0.12, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])  # $/kWh

    energy_system = System(solar_generation, load_demand, price_signal)

    # Create and solve optimization problem
    optimizer = BatteryOptimization(battery, energy_system)
    results = optimizer.solve(initial_soc=25)

    # Plot results
    optimizer.plot_results(results)