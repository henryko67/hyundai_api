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
print(os.getcwd())
os.chdir('/home/henryko67/Projects/selection')

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
