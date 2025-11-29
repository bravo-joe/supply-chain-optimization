import time
import yaml # Read config files
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def query_tbl(config_dir = "config.yaml"):
    # Read the configuration file
    with open(config_dir, "r") as file:
        config = yaml.safe_load(file)
    # Load credentials into variables
    dbname=config["database"]["name"]
    user=config["server"]["user"]
    password=config["server"]["password"]
    host=config["server"]["host"]
    port=config["server"]["port"]
    # Replace with your PostgreSQL credentials and database details
    db_connection_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    # Create the engine
    db_connection = create_engine(db_connection_str)
    # Table name
    tbl_name = 'transportation_problem'
    start_time = time.time() # get start time before insert
    try:
        # Read a whole table
        df_table = pd.read_sql_table(
            tbl_name,
            con=db_connection
        )
        print("Table successfully read from PostgreSQL!")
    except Exception as e:
        print(f"Error reading the table from PostgreSQL database: {e}")
    # Ensure cleanup tasks are performed
    finally: db_connection.dispose()
    end_time = time.time() # get end time after insert
    total_time = end_time - start_time # calculate the time
    print(f"Read time: {total_time:.3f} seconds") # print time
    return(df_table)

# Transportation costs as numpy array
def transportation_costs_arr(df: pd.DataFrame) -> np.ndarray:
    """Turns transportation costs from queried table to a numpy array."""
    return df.iloc[:-1, 1:-1].to_numpy()

# Transportation costs as pandas dataframe
def transportation_costs_df(df: pd.DataFrame) -> pd.DataFrame:
    """Pull transportation costs from queried table to a dataframe."""
    return df.iloc[:-1, 1:-1]      

# Customer demand as numpy array
def extract_demand_arr(df: pd.DataFrame) -> np.ndarray:
    """Pull customer demand, clean data, and output 1d numpy array."""
    demand = df.iloc[-1, 1:]
    demand = demand.reset_index(drop=True)
    demand = demand.dropna()
    demand = demand.astype(int)
    demand = demand.to_numpy()
    return demand

# Customer demand as pandas dataframe
def extract_demand_df(df: pd.DataFrame) -> pd.DataFrame:
    """Pull customer demand, clean data, and output 1d pandas dataframe."""
    demand = df.iloc[-1,:]
    demand = demand.to_frame(name = 'demand')
    demand = demand.dropna()
    demand = demand.iloc[1:, :]
    demand = demand.reset_index()
    demand = demand.drop(labels="index", axis=1)
    return demand

def clean_supply_data(df: pd.DataFrame) -> pd.DataFrame:
    """Data cleaning helper function for production capacity data."""
    prod_cap = df.iloc[:,-1]
    prod_cap = prod_cap.dropna()
    prod_cap = prod_cap.astype(int)
    return prod_cap

def extract_supply_arr(df: pd.DataFrame) -> np.ndarray:
    """Extract the production capacity from each plant into a numpy array."""
    prod_cap = clean_supply_data(df)
    return prod_cap.to_numpy()

def extract_supply_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert supply data from pandas series to dataframe."""
    prod_cap = clean_supply_data(df)
    return prod_cap.to_frame(name='production_capacity')

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
