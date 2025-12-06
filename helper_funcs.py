import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def query_tbl(
    user: str
    , pw: str
    , host: str
    , port: int
    , dbname: str
    , tbl_name: str     
):
    # Replace with your PostgreSQL credentials and database details
    db_connection_str = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{dbname}"
    # Create the engine
    db_connection = create_engine(db_connection_str)
    start_time = time.time() # get start time before insert
    try:
        # Read a whole table
        df_table = pd.read_sql_table(
            tbl_name,
            con=db_connection
        )
        print("Table successfully read from PostgreSQL!")
        end_time = time.time() # get end time after insert
        total_time = end_time - start_time # calculate the time
        print(f"Read time: {total_time:.3f} seconds") # print time
        return(df_table)
    except Exception as e:
        print(f"Error reading the table from PostgreSQL database: {e}")
    # Ensure cleanup tasks are performed
    finally: db_connection.dispose()


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

def migrate_to_rdbms(
        user: str,
        pw: str,
        host: str,
        port: int,
        dbname:str,
        dframe: pd.DataFrame,
        tbl_name: str
):
    """Moves a dataframe to a new table in rdbms."""
    import logging
    # Establish some logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    # Connection to dbase
    db_connection_str = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{dbname}"
    engine = create_engine(db_connection_str)
    start_time = time.time()
    try:
        dframe.to_sql(tbl_name, engine, index=False, if_exists='replace')
        logger.info("DataFrame successfully moved to PostgreSQL!")
        end_time = time.time()
        total_time = end_time - start_time # Calculate the time
        logger.info("Insert time: %.*f seconds", 2, total_time)
    except Exception as e:
        logger.error("Error moving the DataFrame to PostgreSQL database: %s", e)
