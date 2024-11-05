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
from selection import Selector
print(os.getcwd())
os.chdir('/home/henryko67/Projects/selection')

# %%
# import test data
data_path = 'test_all.csv'
# only need 'tag_description'
df = pd.read_csv(data_path, skipinitialspace=True)
df = df[:100]
df = df[['tag_description', 'ships_idx']]
print(len(df))

# %%
##########################################
# run inference
# checkpoint
directory = 'checkpoint_epoch40'
pattern = 'checkpoint-*'
# Use glob to find matching paths
checkpoint_path = glob.glob(os.path.join(directory, pattern))[0]

infer = Inference(checkpoint_path)
infer.prepare_dataloader(df, batch_size=256, max_length=64)
thing_prediction_list, property_prediction_list = infer.generate()


# %%
print(len(thing_prediction_list))
print(len(property_prediction_list))
# %%
