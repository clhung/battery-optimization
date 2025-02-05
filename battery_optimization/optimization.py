from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass
class Battery:
    capacity: float  # kWh
    max_charge: float  # kW
    max_discharge: float  # kW
    charge_efficiency: float = 0.9
    discharge_efficiency: float = 0.9


@dataclass
class System:
    solar: np.ndarray  # kW
    load: np.ndarray  # kW
    price: np.ndarray  # $/kWh

    @property
    def time_periods(self):
        return len(self.load)


class BatteryOptimization:
    def __init__(self, battery: Battery, system: System):
        self.battery = battery
        self.system = system
        self.t = system.time_periods

        # define variables and initiate constraint
        self.charge = cp.Variable(self.t, nonneg=True)
        self.discharge = cp.Variable(self.t, nonneg=True)
        self.soc = cp.Variable(self.t + 1, nonneg=True)
        self.grid_import = cp.Variable(self.t, nonneg=True)
        self.z = cp.Variable(self.t)  # charge/discharge indicator
        self.constraints = []

    def define_constraints(self, initial_soc: float):
        # soc limits
        self.constraints += [self.soc[0] == initial_soc]
        self.constraints += [self.soc <= self.battery.capacity]

        for t in range(self.t):
            # energy balance
            self.constraints += [
                self.soc[t + 1]
                == self.soc[t]
                + self.charge[t] * self.battery.charge_efficiency
                - self.discharge[t] / self.battery.discharge_efficiency
            ]

            # charging/discharging limits
            self.constraints += [
                self.charge[t] <= self.battery.max_charge * self.z[t]
            ]
            self.constraints += [
                self.discharge[t]
                <= self.battery.max_discharge * (1 - self.z[t])
            ]

            # meet load demand
            self.constraints += [
                self.grid_import[t]
                + self.system.solar[t]
                + self.discharge[t] / self.battery.discharge_efficiency
                - self.charge[t] * self.battery.charge_efficiency
                == self.system.load[t]
            ]

    def solve(self, initial_soc: float) -> pd.DataFrame:
        self.define_constraints(initial_soc=initial_soc)
        obj = cp.Minimize(cp.sum(self.grid_import * self.system.price))
        problem = cp.Problem(obj, self.constraints)
        problem.solve(solver=cp.ECOS)

        return pd.DataFrame(
            data={
                "t": [*range(self.t)],
                "charge": self.charge.value,
                "discharge": self.discharge.value,
                "soc": self.soc.value[0:-1],
                "grid_import": self.grid_import.value,
            }
        )


if __name__ == "__main__":
    battery = Battery(
        capacity=50,
        max_charge=10,
        max_discharge=10,
    )

    solar_generation = np.array(
        [
            0,
            0,
            1,
            3,
            7,
            10,
            12,
            14,
            12,
            10,
            8,
            5,
            3,
            2,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    load_demand = np.array(
        [
            5,
            4,
            6,
            7,
            10,
            15,
            18,
            20,
            22,
            18,
            15,
            12,
            10,
            8,
            7,
            6,
            6,
            5,
            4,
            4,
            4,
            4,
            4,
            5,
        ]
    )
    price_signal = np.array(
        [
            0.10,
            0.10,
            0.10,
            0.12,
            0.15,
            0.18,
            0.20,
            0.25,
            0.30,
            0.35,
            0.30,
            0.28,
            0.25,
            0.22,
            0.20,
            0.18,
            0.15,
            0.12,
            0.10,
            0.10,
            0.10,
            0.10,
            0.10,
            0.10,
        ]
    )

    system = System(solar_generation, load_demand, price_signal)

    # Create and solve optimization problem
    optimizer = BatteryOptimization(battery, system)
    results = optimizer.solve(initial_soc=10)
