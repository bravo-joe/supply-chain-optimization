# Beginning of "supply_chain_optimization.py"
"""
    This script writes the directed acyclic graph (DAG) that orchestrates tasks.
    It can create synthetic data that with transporation costs, customers, plants,
    supply, and demand, implement the ACO algorithm, and produce a visual dashboard.
"""
# Import necessary libraries and modules
from pathlib import Path
from datetime import datetime

from airflow.operators.empty import EmptyOperator
from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator

# Store DAG ID in global variable
DAG_ID = Path(__file__).stem

# Define default arguments
default_args = {
    'owner': 'bravo',
    'schedule_interval': None,
    'start_date': datetime(2025, 11, 28),
    'catchup': False,
    'max_active_runs': 1
}

# Instantiate the DAG using decorator
@dag(
    dag_id = DAG_ID,
    default_args = default_args,
    description = "Minimizing transportation costs using ACO.",
    doc_md = """
    ### Orchestrates the tasks from building data to launching dashboard.

    **Steps:**
    1. **Start DAG** - Dummy operator to start the DAG.
    2. **Synthetic Data** - Option of creating and uploading to database data.
    3. **EDA & ACO** - Output tables pertaining to EDA and ACO implementation.
    TO-DO
    4. 
    5. 
    """,
    tags = [
        'operational research',
        'linear optimization',
        'supply chain',
        'ant colony optimization',
        'portfolio project'
    ]
)

def main_dag():
    # First task
    @task
    def start_dag():
        EmptyOperator(task_id="start_the_dag")
    @task
    def intermed_dag():
        print("This is an intermediate task ...")
    @task
    def end_dag():
        EmptyOperator(task_id="end_of_dag")

    # Set task dependencies
    start_dag() >> intermed_dag() >> end_dag()

main_dag()

# End of "supply_chain_optimization.py"