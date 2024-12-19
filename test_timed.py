# %%
import torch
from torch.utils.data import DataLoader
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import numpy as numpy
# import matplotlib.pyplot as plt
import os
import glob
from inference import Inference
import selection
from run_end_to_end import run_end_to_end
import json
print(os.getcwd())
os.chdir('/home/henryko67/Projects/selection')

# %%
# Opening JSON file
with open('output.json') as json_file:
    input = json.load(json_file)


# input = list(input)
# %%
df_list = []
# create combined dataframe such that ships_idx is intruded into the ship_data_list
for ship in input:
    print(ship)
    ships_idx = ship['ships_idx']
    print(ships_idx)

    # i need the data to be a single df
    data = [{
        'index': data['index'],
        'tag_description': data['tag_description'],
        'unit': data['unit'],
        'ships_idx': ships_idx, 
        } 
        for data in ship['ship_data_list']]
    df = pd.DataFrame(data)

    df_list.append(df)

df = pd.concat(df_list, axis=0)
    
###############################
# start of inference + process
# df = run_end_to_end(df)

# %%
# %%
# import test data
data_path = 'test_all.csv'
# only need 'tag_description'
df = pd.read_csv(data_path, skipinitialspace=True)
df = df[:100]
# df = df[['tag_description', 'ships_idx']]
df = df[df['ships_idx'] == 1003].reset_index(drop=True)
print(len(df))

# %%
##########################################
# run inference
df = run_end_to_end(df)
# %%

# import sqlite3
# import os
# 
# def get_cache_db_path(ships_idx):
#     """
#     Generate a cache database file path based on ships_idx.
#     """
#     cache_dir = "cache_files"  # This should match your cache directory
#     return os.path.join(cache_dir, f"cache_ship_{ships_idx}.db")
# 
# def print_cache_contents(ships_idx):
#     """
#     Print the contents of the cache database for a specific ship.
# 
#     Args:
#         ships_idx (int): Ship index for the cache database.
#     """
#     cache_db_path = get_cache_db_path(ships_idx)
# 
#     # Check if the database file exists
#     if not os.path.exists(cache_db_path):
#         print(f"No cache database found for ship {ships_idx} at {cache_db_path}")
#         return
# 
#     print(f"Contents of cache database for ship {ships_idx} ({cache_db_path}):")
#     with sqlite3.connect(cache_db_path) as conn:
#         cursor = conn.execute('''
#             SELECT tag_description, unit, thing, property, frequency FROM cache
#         ''')
#         rows = cursor.fetchall()
# 
#         if rows:
#             print(f"{'Tag Description':<30} {'Unit':<10} {'Thing':<20} {'Property':<20} {'Frequency':<10}")
#             print("-" * 90)
#             for row in rows:
#                 print(f"{row[0]:<30} {row[1]:<10} {row[2]:<20} {row[3]:<20} {row[4]:<10}")
#         else:
#             print("Cache is empty.")
# 
# # Example usage:
# if __name__ == "__main__":
#     # Replace with the specific ship index you want to check
#     ship_index = 1003
#     print_cache_contents(ship_index)

# %%
import sqlite3

# Path to the database
db_path = "cache_files/cache_ship_1003.db"

# Connect to the database
with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    
    # Execute a query to select all rows from the cache table
    cursor.execute("SELECT * FROM cache")
    rows = cursor.fetchall()
    
    # Print the results
    for row in rows:
        print(row)


# %%
