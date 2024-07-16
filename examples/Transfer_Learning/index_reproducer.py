import os

# import sys
# from pathlib import Path

import numpy as np
import pandas as pd

# import seaborn as sns
from botorch.test_functions.synthetic import Hartmann

from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.botorch_wrapper import botorch_function_wrapper

from baybe.utils.random import set_random_seed

set_random_seed(42)

# Configuration
SMOKE_TEST = "SMOKE_TEST" in os.environ
DIMENSION = 3
BATCH_SIZE = 1
N_MC_ITERATIONS = 2 if SMOKE_TEST else 50
N_DOE_ITERATIONS = 2 if SMOKE_TEST else 10
POINTS_PER_DIM = 3 if SMOKE_TEST else 5
NUM_INIT = 2 if SMOKE_TEST else 5

# Define objective and search space
objective = SingleTargetObjective(target=NumericalTarget(name="Target", mode="MIN"))
BOUNDS = Hartmann(dim=DIMENSION).bounds

discrete_params = [
    NumericalDiscreteParameter(
        name=f"x{d}",
        values=np.linspace(lower, upper, POINTS_PER_DIM),
    )
    for d, (lower, upper) in enumerate(BOUNDS.T)
]

task_param = TaskParameter(
    name="Function",
    values=["Test_Function", "Training_Function"],
    active_values=["Test_Function"],
)

parameters = [*discrete_params, task_param]
searchspace = SearchSpace.from_product(parameters=parameters)

# Define test functions
test_functions = {
    "Test_Function": botorch_function_wrapper(Hartmann(dim=DIMENSION)),
    "Training_Function": botorch_function_wrapper(
        Hartmann(dim=DIMENSION, negate=True, noise_std=0.15)
    ),
}

# Generate lookup tables
grid = np.meshgrid(*[p.values for p in discrete_params])
lookups = {}
for function_name, function in test_functions.items():
    lookup = pd.DataFrame({f"x{d}": grid_d.ravel() for d, grid_d in enumerate(grid)})
    lookup["Target"] = lookup.apply(function, axis=1)
    lookup["Function"] = function_name
    lookups[function_name] = lookup
lookup_training_task = lookups["Training_Function"]
lookup_test_task = lookups["Test_Function"]

# Perform the transfer learning campaign
campaign = Campaign(searchspace=searchspace, objective=objective)
initial_data = lookup_training_task.sample(n=NUM_INIT)

campaign.add_measurements(initial_data)

df = campaign.recommend(batch_size=BATCH_SIZE)

# print(df)
#         x0   x1   x2       Function
# index
# 40     1.0  0.0  1.0  Test_Function

# NOTE: Indices other than the ones from df are ignored
df["Target"] = lookup_test_task.iloc[df.index]["Target"]

campaign.add_measurements(df)


# NOTE: with SMOKE_TEST and 42 as seed:
# print(lookup_test_task)
#      x0   x1   x2    Target       Function
# 0   0.0  0.0  0.0 -0.067974  Test_Function
# 1   0.0  0.0  0.5 -0.136461  Test_Function
# 2   0.0  0.0  1.0 -0.091332  Test_Function
# 3   0.5  0.0  0.0 -0.097108  Test_Function
# 4   0.5  0.0  0.5 -0.185407  Test_Function
# 5   0.5  0.0  1.0 -0.090204  Test_Function
# 6   1.0  0.0  0.0 -0.030955  Test_Function
# 7   1.0  0.0  0.5 -0.072904  Test_Function
# 8   1.0  0.0  1.0 -0.084769  Test_Function
# 9   0.0  0.5  0.0 -0.018048  Test_Function
# 10  0.0  0.5  0.5 -0.839061  Test_Function
# 11  0.0  0.5  1.0 -1.994263  Test_Function
# 12  0.5  0.5  0.0 -0.025729  Test_Function
# 13  0.5  0.5  0.5 -0.628022  Test_Function
# 14  0.5  0.5  1.0 -1.957039  Test_Function
# 15  1.0  0.5  0.0 -0.008194  Test_Function
# 16  1.0  0.5  0.5 -0.225915  Test_Function
# 17  1.0  0.5  1.0 -1.826650  Test_Function
# 18  0.0  1.0  0.0 -0.000274  Test_Function
# 19  0.0  1.0  0.5 -2.262308  Test_Function
# 20  0.0  1.0  1.0 -0.334829  Test_Function
# 21  0.5  1.0  0.0 -0.000204  Test_Function
# 22  0.5  1.0  0.5 -1.485659  Test_Function
# 23  0.5  1.0  1.0 -0.325958  Test_Function
# 24  1.0  1.0  0.0 -0.000038  Test_Function
# 25  1.0  1.0  0.5 -0.224631  Test_Function
# 26  1.0  1.0  1.0 -0.300476  Test_Function
