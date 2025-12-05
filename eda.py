# Beginning of "eda.py"
import pandas as pd
# Import custom helper functions
from .helper_funcs import (
    extract_demand_df
    , extract_supply_df
    , transportation_costs_df
    , migrate_to_rdbms
)

# OOP
class EDA:
    """Take a random sample and supply/demand tables then push to database."""

    def __init__(
        self,
        user,
        pw,
        host,
        port,
        dbname, 
    ):
        self.user = user
        self.pw = pw
        self.host = host
        self.port = port
        self.dbname = dbname

    def create_random_sample(self, df, n):
        """Take a simple sample for visualizations."""
        trans_costs = transportation_costs_df(df)
        num_of_plants = len(trans_costs.index)
        num_of_customers = len(trans_costs.columns)
        basic_sample = trans_costs.sample(
            n = n,
            axis = 1
        )
        migrate_to_rdbms(
            user = self.user,
            pw = self.pw,
            host = self.host,
            port = self.port,
            dbname = self.dbname,
            dframe = basic_sample,
            tbl_name = "random_sample"
        )

    def join_supply_and_demand(self, df):
        """Extract supply and demand then push back to database."""
        demand = extract_demand_df(df) # Demand
        demand = demand.astype(int) # Convert to native Python int type
        prod_cap = extract_supply_df(df) # Supply
        prod_cap = prod_cap.astype(int)
        supply_and_demand = pd.concat(
            [prod_cap, demand],
            axis=1
        )
        migrate_to_rdbms(
            user = self.user,
            pw = self.pw,
            host = self.host,
            port = self.port,
            dbname = self.dbname,
            dframe = supply_and_demand,
            tbl_name = "supply_and_demand"
        )

# End of "eda.py"