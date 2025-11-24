import time
import yaml # Read config files
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

# Transportation costs
def transportation_costs(
    df,
    yield_df=True
):
    """Pull the data in the middle of the queried table that is just transportation costs."""
    if yield_df:
        return df.iloc[:-1, 1:-1]
    else:
        return df.iloc[:-1, 1:-1].to_numpy()

# Pull demand
def extract_demand(
    df,
    yield_df=True
):
    """
    Pull the demand row, the final row in the queried table.
    Can output either a Pandas series or NumPy array.

    Args:
        df (pandas.dataframe): Table queried from Postgres with transportation costs.
        to_dataframe (bool): Determines whether function outputs a pandas dataframe or numpy array. Default is True.

    Returns:
        Customer demand in one vector (pandas.dataframe or numpy array).
    """
    if yield_df:
        # Series of data processing logic that outputs a dataframe
        demand = df.iloc[-1,:]
        demand = demand.to_frame(name = 'demand')
        demand = demand.dropna()
        demand = demand.iloc[1:, :]
        demand = demand.reset_index()
        demand = demand.drop(labels="index", axis=1)
    else:
        # Convert Pandas dataframe to a NumPy series after data cleaning,
        demand = df.iloc[-1, 1:]
        demand = demand.reset_index(drop=True)
        demand = demand.dropna()
        demand = demand.astype(int)
        demand = demand.to_numpy()
    return demand

# Get the supply
def extract_supply(
    df,
    yield_df=True
):
    """
    Extract the production capacity for each plant which is the last column.
    Performing some data cleaning before deciding on array or dataframe.

    Args:
        df (pandas dataframe): Table queried from Postgres with transportation costs.
        to_dataframe (bool): Determines whether function outputs a pandas dataframe or numpy array. Default is True.

    Returns:
        Production capacity one vector (pandas dataframe or numpy array).
    """
    prod_cap = df.iloc[:,-1]
    prod_cap = prod_cap.dropna()
    prod_cap = prod_cap.astype(int)
    if yield_df:
        # Convert from series to dataframe
        prod_cap = prod_cap.to_frame(name = 'production_capacity')
    else:
        # Convert to numpy array
        prod_cap.to_numpy()
    return prod_cap
