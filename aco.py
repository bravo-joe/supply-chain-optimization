# -*- coding: utf-8 -*-
# Beginning of "aco.py"
import yaml
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from helper_funcs.helper_funcs import (
   extract_demand_arr,
   extract_supply_arr,
   transportation_costs_arr
)

# OOP
class TransportationProblem: # Class 1
    """Stores supply, customer demand, and cost arrays and evaluates the solution cost."""

    def __init__(
        self,
        prod_cap: np.ndarray,
        demand: np.ndarray,
        trans_costs: np.ndarray
    ):
        """
        Initialize transportation problem.
        
        Args:
            prod_cap: Array of productiion capacities for each supplier
            demand: Array of demands for each customer
            trans_costs: Matrix of transportation costs (m x n)

        """
        self.prod_cap = prod_cap
        self.demand = demand
        self.trans_costs = trans_costs

        self.m = len(prod_cap) # Number of suppliers/factories
        self.n = len(demand) # Number of destinations/customers

    def cost(self, x: np.ndarray) -> float:
        """
        Compute total transportation cost for a solution.
        
        Args:
            x: Solution matrix (m x n) representating shipment quantities
        """
        return np.sum(x * self.trans_costs)

@dataclass
class ConvergenceData: # Class 2 (DataClass)
    """Stores convergence information for each iteration."""
    iteration: int
    best_cost: float
    avg_cost: float

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            'iteration': self.iteration,
            'best_cost': self.best_cost,
            'avg_cost': self.avg_cost
        }

class AntColony: # Class 3
    """Implement Ant Colony Optimization on the transportation costs problem."""

    def __init__(
        self,
        problem: TransportationProblem,
        n_ants: int = 30,
        n_iterations: int = 100,
        alpha: int = 1,
        beta: int = 2,
        rho: float = 0.1,
        Q: int = 1,
        seed: Optional[int] = 123
    ):
        """
        Initialize the Ant Colony Optimization algorithm.
        
        Args:
            problem: TransportationProblem instance
            n-ants: Number of ants per iteration
            n_iterations: Number of iterations to run
            alpha: Pheromone importance factor
            beta: Heuristic importance factor
            rho: Pheromone evaporation rate
            Q: Pheromone deposit factor
            seed: Random seed for reproducibility (optional)
        """
        self.problem = problem
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        if seed is not None:
            np.random.seed(seed)

        # Initialize pheromone matrix
        self.pheromone = np.ones((problem.m, problem.n))

        # Track best solution
        self.optimal_solution = None
        self.best_cost = float("inf")

        # Convergence tracking
        self.convergence_history: List[ConvergenceData] = []

    # Construct ant solution
    def create_solution(self) -> np.ndarray:
        """
        Find a feasible transportation plan using probabilistic selection.
        
        Returns:
            Solution matrix representing shipment quantities.
        """
        # Matrix dimensions
        m, n = self.problem.m, self.problem.n
        remaining_supply = self.problem.prod_cap.copy()
        remaining_demand = self.problem.demand.copy()
        
        # Begin with a zero matrix
        solution = np.zeros((m, n))

        # Heuristic: Inverse of cost
        eta = 1 / (self.problem.trans_costs + 1e-6)
        
        while remaining_demand.sum() > 0:
            # For each customer
            for j in range(n):
                if remaining_demand[j] <= 0:
                    continue
                
                # Calculate probability for each supplier i to satisfy demand j
                numerators = (self.pheromone[:, j]**self.alpha) * (eta[:, j]**self.beta)
                
                # Zero out suppliers with no supply left
                numerators = np.where(
                    remaining_supply > 0,
                    numerators,
                    0
                )
                # When no feasible supplier
                if numerators.sum() == 0:
                    continue

                # Probabilistic selection
                prob = numerators/numerators.sum()
                i = np.random.choice(range(m), p = prob)
                
                # Assign as much as possible
                amount = min(remaining_supply[i], remaining_demand[j])
                solution[i, j] += amount
                remaining_supply[i] -= amount
                remaining_demand[j] -= amount
        
        return solution
    
    # Update Pheromones
    def update_pheromones(self,  solutions: List[np.ndarray]) -> None:
        """
        Evaporate and deposit pheromone based on solution quality.
        
        Args:
            solutions: List of solutions from current iteration
        """
        # Evaporation
        self.pheromone = (1 - self.rho)
        
        # Deposit pheromones for all solutions. More for optimal.
        for soln in solutions:
            cost = self.problem.cost(soln)
            if cost > 0: # Check no division by zero
                self.pheromone += (self.Q / cost) * (soln > 0)

    def store_convergence_data(
            self,
            iteration: int,
            best_cost: float,
            avg_cost: float
    ) -> None:
        """
        Store convergence data for current iteration.
        
        Args:
            iteration: Current iteration number
            best_cost: Best cost found so far
            avg_cost: Average cost across all ants in current iteration
        """
        conv_data = ConvergenceData(
            iteration = iteration,
            best_cost = best_cost,
            avg_cost = avg_cost
        )
        self.convergence_history.append(conv_data)

    # Run optimization
    def run_optimization(self, verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        Execute the ACO algorithm.

        Args:
            verbose: Whether to print progress (default is False)

        Returns:
            Tuple of (optimal_solution, best_cost)
        """
        for iter in range(self.n_iterations):
            iter_solutions = []
            iter_costs = []
            
            # Generate solutions for all ants
            for _ in range(self.n_ants):
                soln = self.create_solution()
                cost = self.problem.cost(soln)
                
                iter_solutions.append(soln)
                iter_costs.append(cost)
                
                # Update best solution
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.optimal_solution = soln.copy()

            # Update pheromones
            self.update_pheromones(iter_solutions)

            # Store convergence data
            avg_cost = np.mean(iter_costs)
            self.store_convergence_data(
                iteration = iter + 1,
                best_cost = self.best_cost,
                avg_cost = avg_cost
            )

            if verbose:
                print(f"Iteration {iter + 1} / {self.n_iterations}: "
                      f"Best Cost = {self.best_cost:.2f}, "
                      f"Avg Cost = {avg_cost:.2f}"
                )

        return self.optimal_solution, self.best_cost
    
    def get_results(self) -> Dict:
        """
        Return comprehensive optimization results.
        
        Returns:
            Dictionary containing solution, costs, and convergence data
        """
        return {
            'solution': self.optimal_solution,
            'best_cost': self.best_cost,
            'convergence_history': [data.to_dict() for data in self.convergence_history],
            'parameters': {
                'n_ants': self.n_ants,
                'n_iterations': self.n_iterations,
                'alpha': self.alpha,
                'beta': self.beta,
                'rho': self.rho,
                'Q': self.Q
            }
        }
    
    def get_convergence_data(self) -> List[Dict]:
        """
        Get convergence data as list of dictionaries.
        
        Returns:
            List of convergence data for each iteration 
        """
        return [data.to_dict() for data in self.convergence_history]

# Driver logic
def driver_func(dframe: pd.DataFrame) -> pd.DataFrame:
    """
    Driver function for implementation of ACO algorithm on transportation problem.
    
    Args:
        dframe: Main transportation problem table

    Returns:
        pd.DataFrame: Matrix containing convergence history for the run    
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    demand = extract_demand_arr(dframe)
    prod_cap = extract_supply_arr(dframe)
    trans_costs = transportation_costs_arr(dframe)

    # Step 1: Create TransportationProblem instance
    problem = TransportationProblem(prod_cap, demand, trans_costs)

    # Step 2: Run ACO algorithm
    aco = AntColony(problem = problem)
    optimal_solution, best_cost = aco.run_optimization()

    # Step 3: Display results
    results = aco.get_results()
    logger.info(f"Optimal Cost: {results['best_cost']}")
    logger.info(f"Optimal Solution: {results['solution']}")

    # Step 4: Verify constraints
    supply_check = np.allclose(optimal_solution.sum(axis=1), prod_cap)
    demand_check = np.allclose(optimal_solution.sum(axis=0), demand)

    # Constraint Verification
    logger.info(f"Supply constraints satisfied: {supply_check}")
    logger.info(f"Demand constraints satisfied: {demand_check}")

    # Step 5: Access convergence data
    convergence = aco.get_convergence_data()
    logger.info("Optimization Complete!")

    return pd.DataFrame(convergence)

    # Balance supply and demand
    # if total_supply > total_demand:
    #     demand[-1] += (total_supply - total_demand)
    # else:
    #     prod_cap[-1] += (total_demand - total_supply)

# End of "aco.py"