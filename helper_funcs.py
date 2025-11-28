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

def find_solution(
        demand: np.array,
        supply: np.array,
        transportation_costs: np.array,
        alpha: int,
        beta: int,
        m: int,
        n: int,
        pheromone: np.ndarray
) -> np.array:
    """Minimizes the transportation costs using the ant colony optimization algorithm.

    Args:
        demand (numpy.ndarray): 1D array of the customer demand.
        supply (numpy.ndarray): 1D array of production capacities from various plants/factories.
        transportation_costs (numpy.ndarray): Multi-dimentional table of transportation costs.
        alpha (int): Hyperparameter. Pheromone influence.
        beta (int): Hyperparameter. Heuristic influence.
        m (int): Number of plants/factories/rows from the transportation costs table.
        n (int): Number of customers/columns from the transportation costs table.
        pheromone (numpy.ndarray): Initial, multi-dimensional pheromone (m x n) matrix.

    Returns:
        numpy.ndarray: Feasible transportation plan matrix.
    
    """
    remaining_supply = supply.copy()
    remaining_demand = demand.copy()
    # Begin with a zero matrix
    x = np.zeros((m, n))
    # Inverse of cost
    eta = 1/(transportation_costs + 1e-6)
    while remaining_demand.sum() > 0:
        # For each customer
        for j in range(n):
            if remaining_demand[j] <= 0:
                continue
            # Calculate probability for each supplier i to satisfy demand j
            numerators = (pheromone[:, j]**alpha) * (eta[:, j]**beta)
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
            amount = min(
                remaining_supply[i],
                remaining_demand[j]
            )
            x[i][j] += amount
            remaining_supply[i] -= amount
            remaining_demand[j] -= amount
            if remaining_supply[i] == 0:
                pass
    return x

def solution_cost(
        solution: np.ndarray,
        transportation_costs: np.ndarray
) -> np.float64:
    """Perform matrix operations to get the cost."""
    return np.sum(solution*transportation_costs)

def update_pheromones(
        optimal_solution: np.ndarray,
        pheromone: np.ndarray,
        rho: float,
        Q: int
) -> np.ndarray:
    """Using some hyperparameters and calculations to update pheromones.
    
    Args:
        optimal_solution (numpy.ndarray): Best solution after every iteration.
        pheromone (numpy.ndarray): Reinforce good paths while poorer ones fade away.
        rho (float): Evaporation rate.
        Q (int): Pheromone deposit scaling.

    Returns:
        numpy.ndarray: Update pheromone trails for ants to follow.
    
    """
    pheronome = (1 - rho) * pheromone
    # Deposit on all edges
    pheromone += Q/(solution_cost(optimal_solution) + 1e-6)
    return pheromone
