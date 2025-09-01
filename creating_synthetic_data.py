# Beginning of "creating_synthetic_data.py"
# Import necessary libraries and packages

# Logic to create a list of customers
prefix = "customer-"
numbered_customers = []
for i in range(1, 126): # For loop to generate suffixes from 1 to 125
    numbered_customers.append(f"{prefix}{i}")
print(numbered_customers) # TO-DO: Remove this line in actual code.

num_factories = int((3/5)*len(numbered_customers))




# End of "creating_synthetic_data.py"