# Beginning of "creating_synthetic_data.py"
# Import necessary libraries and packages
import random # For random number generator
import psycopg2 # To connect with PostgreSQL
import yaml # Read config files
import time
import pandas as pd # Dataframes
import numpy as np # Calculations and processing

# Will bundle the different cells in my notebook into one function
def my_function(
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
    max_demand : int
        The maximum a customer can demand in units (default is 301)        
    min_demand : int
        The minimum

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
    num_factories = int((3/5)*len(num_customers))

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
    possible_demand = possible_nums.tolist()
    # Select a random number from this array
    demand = []
    for i in range(1, num_customers+1): # Generates demand for all 75 factories
        #demand.append(np.random.choice(possible_nums))
        demand.append(random.choice(possible_nums))
        print(demand)    

# End of "creating_synthetic_data.py"