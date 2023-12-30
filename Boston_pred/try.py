import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from ssa import ssa  # Assuming you have the SSA algorithm implemented

# Load the Boston housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function for GBDT parameter optimization
def gbdt_objective_function(params):
    n_estimators, learning_rate, min_samples_split, subsample, max_depth = params

    # Initialize and train GBDT model
    gbdt_model = GradientBoostingRegressor(
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        min_samples_split=int(min_samples_split),
        subsample=subsample,
        max_depth=int(max_depth),
        random_state=42
    )
    gbdt_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = gbdt_model.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)

    return mse

# Define the search space for SSA
dim = 5  # Number of parameters to optimize
bounds = [(50, 200), (0.01, 0.1), (2, 20), (0.8, 1.0), (3, 10)]  # Search space for each parameter

# Run SSA to optimize GBDT parameters
best_solution, _ = ssa(gbdt_objective_function, dim, bounds, max_iter=100)

# Extract the best parameters
best_n_estimators, best_learning_rate, best_min_samples_split, best_subsample, best_max_depth = best_solution

# Train the final GBDT model with the best parameters
final_gbdt_model = GradientBoostingRegressor(
    n_estimators=int(best_n_estimators),
    learning_rate=best_learning_rate,
    min_samples_split=int(best_min_samples_split),
    subsample=best_subsample,
    max_depth=int(best_max_depth),
    random_state=42
)
final_gbdt_model.fit(X_train, y_train)

# Plot the optimization process (Optional)
# You can modify this part based on your specific needs
# This is just a basic example to show how the optimization process evolves

# Initialize an array to store the best fitness values at each iteration
best_fitness_values = []

# Run SSA algorithm and collect best fitness values
best_solution, fitness_values = ssa(gbdt_objective_function, dim, bounds, max_iter=10)
best_fitness_values.append(fitness_values)

print("最优解:", best_solution)
print("最优解对应的目标函数值:", fitness_values)

best_solution = [1.54725612e+02, 8.71091040e-02, 8.19613136e+00, 9.38370398e-01, 5.47840556e+00]

best_n_estimators, best_learning_rate, best_min_samples_split, best_subsample, best_max_depth = best_solution


