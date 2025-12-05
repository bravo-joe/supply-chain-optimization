# Beginning of "supply_chain_optimization.py"
"""
    This script writes the directed acyclic graph (DAG) that orchestrates tasks.
    It can create synthetic data that with transporation costs, customers, plants,
    supply, and demand, implement the ACO algorithm, and produce a visual dashboard.
"""
# Import necessary libraries and modules
import yaml # Read config files 
from pathlib import Path
from datetime import datetime
import os

from airflow.operators.empty import EmptyOperator
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.python import PythonOperator

# Custom functions
# from helper_funcs.latin import some_latin
from helper_funcs.creating_synthetic_data import (
    create_transportation_costs
    , migrate_trans_costs_tbl
)
from helper_funcs.helper_funcs import query_tbl
from helper_funcs.eda import EDA

# Define global variables and read configuration file
DAG_ID = Path(__file__).stem
config_path = os.path.join(os.path.dirname(__file__), "helper_funcs/config.yaml")
# with open(config_path, "r") as file:
with open(config_path, "r") as file:
    CONFIG = yaml.safe_load(file)
DBNAME = CONFIG["database"]["name"]
TABLE = CONFIG["database"]["table"]
USER = CONFIG["server"]["user"]
PASSWORD = CONFIG["server"]["password"]
HOST = CONFIG["server"]["host"]
PORT = CONFIG["server"]["port"]
N_CUSTOMERS = CONFIG["company_data"]["num_customers"]
MIN_COST = CONFIG["company_data"]["min_cost"]
MAX_COST = CONFIG["company_data"]["max_cost"]
CREATE_DATA = CONFIG["synthetic_data"]["necessity"]

# Define default arguments
default_args = {
    'owner': 'bravo'
    , 'schedule_interval': None
    , 'start_date': datetime(2025, 11, 28)
    , 'catchup': False
    , 'max_active_runs': 1
}

# Instantiate the DAG using decorator
@dag(
    dag_id = DAG_ID
    , default_args = default_args
    , description = "Minimizing transportation costs using ACO."
    , doc_md = """
    ### Orchestrates the tasks from building data to launching dashboard.

    **Steps:**
    1. **Start DAG** - Dummy operator to start the DAG.
    2. **Synthetic Data** - Option of creating and uploading to database data.
    3. **EDA & ACO** - Output tables pertaining to EDA and ACO implementation.
    4. 
    5. 
    """
    , tags = [
        'supply chain optimization'
        , 'portfolio project'
        , 'manual_trigger'
    ]
)

def main_dag():
    
    # First task
    @task
    def start_dag():
        EmptyOperator(task_id="start_dag")
    
    @task
    def create_data():
        if CREATE_DATA:
            migrate_trans_costs_tbl(
                func = create_transportation_costs
                , num_customers = N_CUSTOMERS
                , min_cost = MIN_COST
                , max_cost = MAX_COST
                , dbname = DBNAME
                , tbl = TABLE
                , user = USER
                , password = PASSWORD
                , host = HOST
                , port = PORT
            )
        else:
            pass
    
    @task
    def intermed_task():
        print("This is an intermediate task ...")

    @task
    def perform_eda():
        df = query_tbl(
            user = USER
            , pw = PASSWORD
            , host = HOST
            , port = PORT
            , dbname = DBNAME
            , tbl_name = TABLE
        )   
        eda = EDA(
            user = USER
            , pw = PASSWORD
            , host = HOST
            , port = PORT
            , dbname = DBNAME
        )
        eda.create_random_sample(df = df, n = 15)
        eda.join_supply_and_demand(df = df)

    @task
    def final_task():
        EmptyOperator(task_id="final_task")

    # Set task dependencies
    start_dag() >> [create_data(), intermed_task()] >> perform_eda() >> final_task()

main_dag()

# End of "supply_chain_optimization.py"