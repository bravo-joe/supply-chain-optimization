# Beginning of "eda.py"
# Import necessary libraries and modules
import yaml # Read config files
import os
import pandas as pd
pd.set_option('display.max_rows', 12)
pd.set_option('display.max_columns', 10)
from helper_funcs import (
    query_tbl
    , extract_demand
    , extract_supply
    , transportation_costs
)

# Import table from PostgreSQL
df = query_tbl()
# Gathering just transportation costs
trans_costs = transportation_costs(df)
# Gathering some basic info
num_of_plants = len(trans_costs.index)
num_of_customers = len(trans_costs.columns)
# Want a random subset of 15 customers
n_cols = 15
sample15 = trans_costs.sample(
    n = n_cols,
    axis = 1
)
sample15.to_csv('./data/sample15.csv', index=False)

# Next is the dataframe for customer demand
# Extract the demand row into its own Pandas Series
demand = extract_demand(df)

# Supply
prod_cap = extract_supply(df)

# Combine the two dataframes
eda_df = pd.concat(
    [
        prod_cap,
        demand
    ],
    axis=1
)

eda_df.to_csv('./data/eda_df.csv', index=False)
print(eda_df)

# End of "eda.py"