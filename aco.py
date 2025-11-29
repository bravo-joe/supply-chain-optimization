#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from helper_funcs import (
    query_tbl,
    transportation_costs_arr,
    extract_demand_arr,
    extract_supply_arr,
    TransportationProblem,
    AntColony
)

def main():
    """Main script to run the ACO algorithm."""
    # problem dimensions
    # m=75 # num of suppliers
    # n = 125 # num of destinations

    # Generate sample data
    # print("Generating transportation problem data...")
    # print(f"Suppliers: {m}, Customers: {n}\n")

    # production capacity (supply) for each factory
    # prod_cap = np.random.randint(50, 150, m)
    # Demand for each customer
    # demand = np.random.randint(20, 80, n)

    # Query table from database
    df = query_tbl()
    # --------------------------------------------------------------
    # Extracting customer demand, supply, and transportation costs
    # --------------------------------------------------------------
    demand = extract_demand_arr(df)
    prod_cap = extract_supply_arr(df)
    trans_costs = transportation_costs_arr(df)

    # Setting ACO hyperparameters as global variables 
    n_ants = 30
    n_iterations = 100
    alpha = 1
    beta = 2
    rho = 0.1
    Q = 1
    m, n = trans_costs.shape # Size of transportation costs matrix
    seed = 123

    # Balance supply and demand
    total_supply = prod_cap.sum()
    total_demand = demand.sum()

    # if total_supply > total_demand:
    #     demand[-1] += (total_supply - total_demand)
    # else:
    #     prod_cap[-1] += (total_demand - total_supply)

    print(f"Total supply: {prod_cap.sum()}")
    print(f"Total demand: {demand.sum()}")

    # Transportation costs matrix
    # trans_costs = np.random.rand(m, n)*100

    # Step 1: Create TransportationProblem instance
    print("\n" + "="*70)
    print("Creating Transportation Problem Instance")
    print("="*70)
    problem = TransportationProblem(prod_cap, demand, trans_costs)
    print(f"Problem created with {problem.m} suppliers and {problem.n} customers")
    
    # Step 2: Create AntColony instance
    print("\n" + "="*70)
    print("Initializing Ant Colony Optimizer")
    print("="*70)
    aco = AntColony(
        problem=problem,
        n_ants = n_ants,
        n_iterations = n_iterations,
        alpha = alpha,
        beta = beta,
        rho = rho,
        Q = Q,
        seed = seed
    )
    print(f"ACO initialized with {aco.n_ants} ants and {aco.n_iterations} iterations")
    
    # Step 3: Run optimization
    print("\n" + "="*70)
    print("Running Optimization")
    print("="*70 + "\n")
    
    optimal_solution, best_cost = aco.run_optimization()
    
    # Step 4: Display results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"\nMinimum Transportation Cost: ${best_cost:.2f}")
    print(f"\nOptimal Solution Shape: {optimal_solution.shape}")
    print(f"Non-zero allocations: {np.count_nonzero(optimal_solution)}")
    
    # Show sample of the solution
    print("\nSample of Optimal Transportation Plan (first 5x5):")
    print(optimal_solution[:5, :5])
    
    # Verify constraints
    supply_check = np.allclose(optimal_solution.sum(axis=1), prod_cap)
    demand_check = np.allclose(optimal_solution.sum(axis=0), demand)
    
    print(f"\nConstraint Verification:")
    print(f"  Supply constraints satisfied: {supply_check}")
    print(f"  Demand constraints satisfied: {demand_check}")
    
    print("\n" + "="*70)
    print("Optimization Complete!")
    print("="*70)
    
    return optimal_solution, best_cost, problem, aco

if __name__ == "__main__":
    optimal_solution, best_cost, problem, aco = main()

# End of "main2.py"