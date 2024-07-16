import os

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from botorch.test_functions.synthetic import Hartmann

from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.botorch_wrapper import botorch_function_wrapper

from baybe.utils.random import set_random_seed

seed = 42
set_random_seed(seed)

# Configuration
SMOKE_TEST = "SMOKE_TEST" in os.environ
DIMENSION = 3
BATCH_SIZE = 1
N_MC_ITERATIONS = 2 if SMOKE_TEST else 50
N_DOE_ITERATIONS = 2 if SMOKE_TEST else 10
POINTS_PER_DIM = 3 if SMOKE_TEST else 5
NUM_INIT = 2 if SMOKE_TEST else 10

# Define objective and search space
obj_name = "Target"
mode = "MIN"
objective = SingleTargetObjective(target=NumericalTarget(name=obj_name, mode=mode))
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

# Generate lookup tables (i.e., predefined candidates with corresponding target values)
grid = np.meshgrid(*[p.values for p in discrete_params])
lookups = {}
for function_name, function in test_functions.items():
    lookup = pd.DataFrame({f"x{d}": grid_d.ravel() for d, grid_d in enumerate(grid)})
    lookup[obj_name] = lookup.apply(function, axis=1)
    lookup["Function"] = function_name
    lookups[function_name] = lookup
lookup_training_task = lookups["Training_Function"]
lookup_test_task = lookups["Test_Function"]

# Perform the transfer learning campaign
campaign = Campaign(searchspace=searchspace, objective=objective)
initial_data = lookup_training_task.sample(n=NUM_INIT)

parameter_names = [p.name for p in searchspace.parameters]

full_lookup = pd.concat([lookup_training_task, lookup_test_task], ignore_index=True)

campaign.add_measurements(initial_data)

for i in range(N_DOE_ITERATIONS):

    df = campaign.recommend(batch_size=BATCH_SIZE)

    # update the dataframe with the target value(s)
    df = pd.merge(
        df, full_lookup[parameter_names + [obj_name]], on=parameter_names, how="left"
    )

    campaign.add_measurements(df)
    print("")
    print("Added new measurement(s):")
    print(df)

measurements = campaign.measurements
train_measurements = measurements[measurements["Function"] == "Training_Function"]
test_measurements = measurements[measurements["Function"] == "Test_Function"]

fig, axes = plt.subplots(figsize=(6, 4), dpi=120)

# Plot training measurements
axes.plot(
    train_measurements.index,
    train_measurements[obj_name],
    ls="None",
    marker="o",
    mfc="None",
    mec="gray",
    mew=1,
    label="Training Observed",
)

assert mode == "MIN"

# Plot test measurements
axes.plot(
    test_measurements.index,
    test_measurements[obj_name],
    ls="None",
    marker="o",
    mfc="None",
    mec="k",
    label="Test Observed (↓=better)",
)

# Best to trial only for test measurements
best_to_trial = test_measurements[obj_name].cummin()
axes.plot(
    test_measurements.index, best_to_trial, color="#0033FF", lw=2, label="Best to Trial"
)

plt.xticks(range(len(measurements)))
plt.xlabel("Batch Number")

plt.ylabel(f"{obj_name}")

axes.axvline(
    x=test_measurements.index[0], color="k", linestyle="--", label="Switch Task"
)

plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
dir = "examples/Transfer_Learning/plots/basic-tl"
Path(dir).mkdir(parents=True, exist_ok=True)
fname_stem = f"num_train={NUM_INIT}_num_iter={N_DOE_ITERATIONS}_seed={seed}"
if SMOKE_TEST:
    fname_stem += "_smoke_test"
fname = f"{fname_stem}.png"
plt.savefig(os.path.join(dir, fname), bbox_inches="tight")

1 + 1
