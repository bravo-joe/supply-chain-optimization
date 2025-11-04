# Beginning of "eda.py"
# Import necessary libraries and modules
import yaml # Read config files
import os
import pandas as pd
pd.set_option('display.max_rows', 12)
pd.set_option('display.max_columns', 10)
from helper_funcs import read_tbl

# Import table from PostgreSQL
df = read_tbl()

# Gathering just transportation costs
trans_costs = df.iloc[:-1, 1:-1]

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
print(sample15)
# End of "eda.py"