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
print(os.getcwd())
os.chdir('/home/henryko67/Projects/selection')

# %%
# import test data
data_path = 'test_all.csv'
# only need 'tag_description'
df = pd.read_csv(data_path, skipinitialspace=True)
df = df[:100]
df = df[['tag_description', 'ships_idx']]
df = df[df['ships_idx'] == 1003].reset_index(drop=True)
print(len(df))

# %%
##########################################
# run inference
# checkpoint
directory = 'checkpoint_fold_1'
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

# %%
# add labels too
# thing_actual_list, property_actual_list = decode_preds(pred_labels)
# Convert the list to a Pandas DataFrame
df_out = pd.DataFrame({
    'p_thing': thing_prediction_list, 
    'p_property': property_prediction_list
})
# df_out['p_thing_correct'] = df_out['p_thing'] == df_out['thing']
# df_out['p_property_correct'] = df_out['p_property'] == df_out['property']
df = pd.concat([df, df_out], axis=1)


# %%
# we start to cull predictions from here
data_master_path = f"data_files/data_model_master_export.csv"
df_master = pd.read_csv(data_master_path, skipinitialspace=True)
data_mapping = df
# Generate patterns    
df_master['master_pattern'] = df_master['thing'] + " " + df_master['property']    
# Create a set of unique patterns from master for fast lookup    
master_patterns = set(df_master['master_pattern'])
thing_patterns = set(df_master['thing'])

# %%
# check if prediction is in MDM
data_mapping['p_thing_pattern'] = data_mapping['p_thing'].str.replace(r'\d', '#', regex=True)
data_mapping['p_property_pattern'] = data_mapping['p_property'].str.replace(r'\d', '#', regex=True)
data_mapping['p_pattern'] = data_mapping['p_thing_pattern'] + " " + data_mapping['p_property_pattern']
data_mapping['p_MDM'] = data_mapping['p_pattern'].apply(lambda x: x in master_patterns)    

df = data_mapping

# %%
# %%
# get target data
data_path = "data_files/train.csv"
train_df = pd.read_csv(data_path, skipinitialspace=True)
# processing to help with selection later
train_df['thing_property'] = train_df['thing'] + " " + train_df['property']

# %%
old_thing_pred_list, old_property_pred_list = thing_prediction_list, property_prediction_list
# %%
import importlib
import selection
importlib.reload(selection)

selector = selection.Selector(input_df=df, reference_df=train_df)
thing_prediction_list, property_prediction_list = selector.run_selection(checkpoint_path=checkpoint_path)

##########################################
# %%
# %%
df
# %%
# print(property_prediction_list)
# %%
thing_prediction_list

# %%
property_prediction_list

# %%
old_thing_pred_list
# %%
cat_list = [ old_thing_pred_list[idx] + '_' + old_property_pred_list[idx] for idx in range(len(thing_prediction_list))]
# %%
cat_list
# %%
len(cat_list) - len(set(cat_list))

# %%
# %%
