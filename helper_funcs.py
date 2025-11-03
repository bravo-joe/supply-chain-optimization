import yaml # Read config files
import time
import pandas as pd
from sqlalchemy import create_engine

def read_tbl(config_dir = "config.yaml"):
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