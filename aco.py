# -*- coding: utf-8 -*-
# Beginning of "aco.py"
import yaml
import numpy as np
from helper_funcs import (
    query_tbl,
    transportation_costs_arr,
    extract_demand_arr,
    extract_supply_arr,
    TransportationProblem,
    AntColony
)

# OOP
class TransportationProblem:
    """Stores supply, customer demand, and cost arrays and evaluates the solution cost."""

    def __init__(
        self,
        prod_cap,
        demand,
        trans_costs
    ):
        self.prod_cap = prod_cap
        self.demand = demand
        self.trans_costs = trans_costs

        self.m = len(prod_cap) # Number of suppliers/factories
        self.n = len(demand) # Number of destinations/customers

    def cost(self, x):
        """Return transportation costs."""
        return np.sum(x * self.trans_costs)
    
class AntColony:
    """Implement Ant Colony Optimization on the transportation costs problem."""

    def __init__(
        self,
        problem: TransportationProblem,
        n_ants = 30,
        n_iterations = 100,
        alpha = 1,
        beta = 2,
        rho = 0.1,
        Q = 1,
        seed = 123
    ):
        self.problem = problem
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        np.random.seed(seed)

        # Initialize pheromone matrix
        self.pheromone = np.ones((problem.m, problem.n))

        # Track best solution
        self.optimal_solution = None
        self.best_cost = float("inf")
        self.cost_history = []

    # Construct ant solution
    def create_solution(self):
        """Find a feasible transportation plan using probabilistic selection."""
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
                prob = numerators/numerators.sum()
                i = np.random.choice(
                    range(m),
                    p = prob
                )
                # Assign as much as possible
                amount = min(remaining_supply[i], remaining_demand[j])
                solution[i][j] += amount
                remaining_supply[i] -= amount
                remaining_demand[j] -= amount
        return solution
    
    # Update Pheromones
    def update_pheromones(
            self,
            solution
    ):
        """Evaporate and deposit pheromone based on solution quality."""
        cost = self.problem.cost(solution)
        # Evaporation
        self.pheromone = (1 - self.rho) * self.pheromone
        # Deposit
        self.pheromone += self.Q/(cost + 1e-6)

    # Run optimization
    def run_optimization(self):
        for iter in range(self.n_iterations):
            for _ in range(self.n_ants):
                solution = self.create_solution()
                cost = self.problem.cost(solution)
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.optimal_solution = solution
            self.update_pheromones(self.optimal_solution)
            self.cost_history.append(self.best_cost)
            print(f"Iteration {iter + 1}, best cost =  {self.best_cost}")
        return self.optimal_solution, self.best_cost
    
    def get_results(self):
        """Return the best solution and cost."""
        return {
            'solution': self.optimal_solution,
            'cost': self.best_cost,
            'cost_history': self.cost_history
        }

# Now, implement the ACO algorithm by instantiating classes above.
with open("config.yaml", "r") as file:
    CONFIG = yaml.safe_load(file)
# Put database login creds in a list
creds = {
    'USER': CONFIG["server"]["user"],
    'PASSWORD': CONFIG["server"]["password"],
    'HOST':  CONFIG["server"]["host"],
    'PORT': CONFIG["server"]["port"],
    'DBNAME': CONFIG["database"]["name"],
    'TABLE': CONFIG["database"]["table"],
}

# Put the default ACO hyperparameters in a dictionary
default_hyper_params = {
    'n_ants': 30,
    'n_iterations': 100,
    'alpha': 1,
    'beta': 2,
    'rho': 0.1,
    'Q': 1,

}

def implement_aco(
        creds: dict,
        hyper_params: dict,
        tbl_name = "transportation_problem"
):
    """Main script to run the ACO algorithm."""
    df = query_tbl(
        creds['USER'],
        creds['PASSWORD'],
        creds['HOST'],
        creds['PORT'],
        creds['DBNAME'],
        tbl_name,
    )
    # --------------------------------------------------------------
    # Extracting customer demand, supply, and transportation costs
    # --------------------------------------------------------------
    demand = extract_demand_arr(df)
    prod_cap = extract_supply_arr(df)
    trans_costs = transportation_costs_arr(df)

    # Setting ACO hyperparameters as global variables 

    m, n = trans_costs.shape # Size of transportation costs matrix

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
        problem = problem,
        n_ants = hyper_params['n_ants'],
        n_iterations = hyper_params['n_iterations'],
        alpha = hyper_params['alpha'],
        beta = hyper_params['beta'],
        rho = hyper_params['rho'],
        Q = hyper_params['Q'],
        seed = hyper_params['seed'],
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

# End of "aco.py"