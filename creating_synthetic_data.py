# Beginning of "creating_synthetic_data.py"
# Import necessary libraries and packages
import random # For random number generator
import psycopg2 # To connect with PostgreSQL
import yaml # Read config files
import time
import pandas as pd # Dataframes
import numpy as np # Calculations and processing
from sqlalchemy import create_engine

# Will bundle the different cells in my notebook into one function
def create_transp_cost_df(
        num_customers: int,
        min_cost: int = 1,
        max_cost: int = 15,
    ) -> pd.DataFrame:
    """Create a table that contains transportation costs, demand, and manufacturing capacity.

    Parameters
    ----------
    num_customers : int
        The number of customers the company serves
    min_cost : int
        The minimum transportation cost in dollars (default is 1)
    max_cost : int
        The maximum transportation cost for a customer in dollars (default is 15)

    Returns
    -------
    pd.DataFrame
        A matrix made of pseudo-random transportation costs, demand, and production capabilities
    """
    # Create customers
    prefix = "customer_"
    numbered_customers = []
    for i in range(1, num_customers+1): # For loop to generate suffixes from 1 to 125
        numbered_customers.append(f"{prefix}{i}")

    # Make factories list 3/5 of the customers list
    num_factories = int((3/5)*num_customers)

    # Create a number of fictitious factories
    prefix_2 = "factory_"
    numbered_factories = []
    for i in range(1, num_factories+1): # Generates suffixes from 1 to 125
        numbered_factories.append(f"{prefix_2}{i}")
    
    # Random numpy array then dataframe
    random_array = np.random.randint(
        min_cost,
        max_cost+1,
        (
            num_factories,
            num_customers
        )
    )
    random_matrix_df = pd.DataFrame(random_array)

    # Add plant list as index to the pandas df
    random_matrix_df.index = numbered_factories
    # Similarly, turn list of customers to the columns in the df
    random_matrix_df.columns = numbered_customers

    # Array to chose the possible demand by customer
    possible_demand = np.arange(50, 301, 10)
    possible_demand = possible_demand.tolist()
    # Select a random number from this array
    demand = []
    for i in range(1, num_customers+1): # Generates demand for all 75 factories
        #demand.append(np.random.choice(possible_nums))
        demand.append(random.choice(possible_demand))
    
    # List of manufacturing capacity
    list_manufac_cap = [
        125,
        250,
        375,
        500,
        625,
        750
    ]
    
    # Create a column to represent production capacity
    prod_capacity = []
    for j in range(1, num_factories+1): # Generates demand for all 75 factories
        prod_capacity.append(random.choice(list_manufac_cap))

    # Need a None/NaN for the intersection of production capacity and demand
    prod_capacity.append(None)

    # Add the demand row created to the matrix
    random_matrix_df.loc['demand'] = demand

    # Add a the production capacity as a new column with values from a list
    random_matrix_df['production_capacity'] = prod_capacity

    return random_matrix_df

# Initiate the function to create a matrix
transportation_costs_matrix = create_transp_cost_df(125) # 125 customers

# Connect to database and push table to it
# Read the configuration file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

dbname=config["database"]["name"]
user=config["server"]["user"]
password=config["server"]["password"]
host=config["server"]["host"]
port=config["server"]["port"]

# Inserts PostgreSQL credentials and database details for a connection
db_connection_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
# Create the engine
engine = create_engine(db_connection_str)
start_time = time.time() # get start time before insert
try:
    transportation_costs_matrix.to_sql(config["database"]["table"], engine, index=True, if_exists='replace')
    print("DataFrame successfully moved to PostgreSQL!")
except Exception as e:
    print(f"Error moving the DataFrame to PostgreSQL database: {e}")
end_time = time.time() # get end time after insert
total_time = end_time - start_time # calculate the time
print(f"Insert time: {total_time:.3f} seconds") # print time

# End of "creating_synthetic_data.py"