# Beginning of "eda.py"
# Import necessary libraries and modules
import yaml # Read config files
import os
import pandas as pd
pd.set_option('display.max_rows', 12)
pd.set_option('display.max_columns', 10)
from helper_funcs import query_tbl

# Import table from PostgreSQL
df = query_tbl()
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
# print(sample15)

# Next is the dataframe for customer demand
# Extract the demand row into its own Pandas Series
demand = df.iloc[-1,:]
# Convert from series to dataframe
demand = demand.to_frame(name = 'demand')
# Drop NAs from the dataframe
demand = demand.dropna()
# Some rearranging
demand = demand.iloc[1:, :]
# Resetting index
demand = demand.reset_index()
# Drop unnecessary column that was the index
demand = demand.drop(labels="index", axis=1)

# The last dataframe is the production capacity for the various factories
# Pull the last column which is the production capacity
prod_cap = df.iloc[:,-1]
# Remove NA's
prod_cap = prod_cap.dropna()
# Turn data type from float to integer
prod_cap = prod_cap.astype(int)
# Convert from series to dataframe
prod_cap = prod_cap.to_frame(name = 'production_capacity')

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