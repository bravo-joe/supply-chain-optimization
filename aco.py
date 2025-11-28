#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import necessary libraries
import os
import numpy as np
# Set random seed for reproducibility
np.random.seed(123)
# Custom functions
from helper_funcs import (
    query_tbl,
    transportation_costs_arr,
    extract_demand_arr,
    extract_supply_arr,
    find_solution,
    solution_cost,
    update_pheromones
)
# Move to project directory
project_dir = '/home/joe/Documents/supply-chain-optimization/'
os.chdir(project_dir)
# Query table from database
df = query_tbl()
# --------------------------------------------------------------
# Extracting customer demand, supply, and transportation costs
# --------------------------------------------------------------
demand = extract_demand_arr(df)
supply = extract_supply_arr(df)
trans_costs = transportation_costs_arr(df)

# Setting ACO hyperparameters as global variables 
n_ants = 30
n_iterations = 100
alpha = 1
beta = 2
rho = 0.1
Q = 1
m, n = trans_costs.shape # Size of transportation costs matrix

# Initialize m x n pheromone matrix
pheromone = np.ones((m, n))

# Main ACO loop
optimal_solution = None
best_cost = float('inf')
for iter in range(n_iterations):
    solutions = []
    costs = []
    for ant in range(n_ants):
        x = find_solution(
            demand=demand,
            supply=supply,
            transportation_costs=trans_costs,
            alpha=alpha,
            beta=beta,
            m=m,
            n=n,
            pheromone=pheromone
        )
        c = solution_cost(
            solution=pheromone,
            transportation_costs=trans_costs
        )
        solutions.append(x)
        costs.append(c)
        if c < best_cost:
            best_cost = c
            optimal_solution = x
    update_pheromones(
        transportation_costs=trans_costs,
        optimal_solution=optimal_solution,
        pheromone=pheromone,
        rho=rho,
        Q=Q
    )
    print(f"Iteration {iter+1}, best cost = {best_cost}")
# --------------------------------------------------------------
# Results
# --------------------------------------------------------------
print("\n Best Transportation Plan:")
print(optimal_solution)
print("\nMinimum Cost:", best_cost)


