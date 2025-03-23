from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass
class Battery:
    capacity: float  # kWh
    max_charge: float  # kW
    max_discharge: float  # kW
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95


@dataclass
class System:
    timestamp: np.ndarray
    solar: np.ndarray
    load: np.ndarray
    price: np.ndarray

    @property
    def timesteps(self):
        return len(self.timestamp)


class BatteryOptimization:
    def __init__(self, battery: Battery, system: System):
        self.battery = battery
        self.system = system
        self.ts = system.timesteps
        self.dispatch_events = np.zeros(self.ts)
        self.dispatch_amount = 1000
        self.export_price = np.zeros(self.ts)

    def define_variables(self):
        self.charge = cp.Variable(self.ts, nonneg=True)
        self.discharge = cp.Variable(self.ts, nonneg=True)
        self.soc = cp.Variable(self.ts + 1, nonneg=True)
        self.solar_gen = cp.Variable(self.ts, nonneg=True)
        self.grid = cp.Variable(self.ts)
        # control charge and discharge
        self.z = cp.Variable(self.ts, boolean=True)

    def define_constraints(self, initial_soc: float):
        self.constraints = []

        # battery states
        self.constraints += [
            self.soc[0]
            == initial_soc
            + self.charge[0] * self.battery.charge_efficiency
            - self.discharge[0]
        ]
        self.constraints += [self.soc <= self.battery.capacity]

        for t in range(self.ts):
            self.constraints += [
                self.soc[t + 1]
                == self.soc[t]
                + self.charge[t] * self.battery.charge_efficiency
                - self.discharge[t]
            ]
            self.constraints += [
                self.charge[t] <= self.z[t] * self.battery.max_charge
            ]
            self.constraints += [
                self.discharge[t]
                <= (1 - self.z[t]) * self.battery.max_discharge
            ]

            # power balance
            self.constraints += [
                self.grid[t] + self.solar_gen[t] + self.discharge[t]
                == self.system.load[t]
                + self.charge[t] * self.battery.charge_efficiency
            ]

            # solar availability
            self.constraints += [
                self.charge[t] + self.solar_gen[t] <= self.system.solar[t]
            ]

    def solve(
        self, initial_soc: float, dispatch: bool = False
    ) -> pd.DataFrame:
        self.define_variables()
        self.define_constraints(initial_soc=initial_soc)

        timestamp = pd.to_datetime(self.system.timestamp)
        if dispatch:
            self.dispatch_events[
                (timestamp.hour == 18) | (timestamp.hour == 19)
            ] = 1

        # Stage 1: Dispatch Target
        dispatch_obj = cp.Maximize(
            cp.sum(
                cp.multiply(self.dispatch_events, self.discharge - self.grid)
            )
        )
        problem1 = cp.Problem(dispatch_obj, self.constraints)
        problem1.solve(solver=cp.SCIPY, verbose=True)

        grid_fixed = self.grid.value.copy()

        # Stage 2: Minimize Cost with Dispatch Results from Stage 1
        fix_constraints = [
            self.grid[t] == grid_fixed[t]
            for t in range(self.ts)
            if self.dispatch_events[t] == 1
        ]
        cost_obj = cp.Minimize(
            cp.sum(cp.multiply(cp.pos(self.grid), self.system.price))
        )
        problem2 = cp.Problem(cost_obj, self.constraints + fix_constraints)
        problem2.solve(solver=cp.SCIPY)

        return pd.DataFrame(
            data={
                "timestamp": self.system.timestamp,
                "charge": self.charge.value,
                "discharge": self.discharge.value,
                "soc": self.soc.value[0:-1],
                "grid": self.grid.value,
                "solar_gen": self.solar_gen.value,
            }
        )
