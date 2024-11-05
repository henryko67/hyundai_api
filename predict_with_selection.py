# %%
import torch
from torch.utils.data import DataLoader
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import os
import glob
from inference import Inference
from selection import Selector
print(os.getcwd())
os.chdir('/home/richard/Projects/learn_t5/selection')

# %%
# import test data
data_path = '/home/richard/Projects/06_research/hipom_data_mapping/data_preprocess/dataset/1/test_all.csv'
df = pd.read_csv(data_path, skipinitialspace=True)
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
print(len(thing_prediction_list))
print(len(property_prediction_list))

# %%
# we start to cull predictions from here
data_master_path = f"/home/richard/Projects/06_research/hipom_data_mapping/data_import/data_model_master_export.csv"
df_master = pd.read_csv(data_master_path, skipinitialspace=True)
data_mapping = df
# Generate patterns    
data_mapping['thing_pattern'] = data_mapping['thing'].str.replace(r'\d', '#', regex=True)
data_mapping['property_pattern'] = data_mapping['property'].str.replace(r'\d', '#', regex=True)
data_mapping['pattern'] = data_mapping['thing_pattern'] + " " + data_mapping['property_pattern']
df_master['master_pattern'] = df_master['thing'] + " " + df_master['property']    
# Create a set of unique patterns from master for fast lookup    
master_patterns = set(df_master['master_pattern'])
thing_patterns = set(df_master['thing'])
# Check each pattern in data_mapping if it exists in df_master and assign the "MDM" field    
data_mapping['MDM'] = data_mapping['pattern'].apply(lambda x: x in master_patterns)    

# %%
# check if prediction is in MDM
data_mapping['p_thing_pattern'] = data_mapping['p_thing'].str.replace(r'\d', '#', regex=True)
data_mapping['p_property_pattern'] = data_mapping['p_property'].str.replace(r'\d', '#', regex=True)
data_mapping['p_pattern'] = data_mapping['p_thing_pattern'] + " " + data_mapping['p_property_pattern']
data_mapping['p_MDM'] = data_mapping['p_pattern'].apply(lambda x: x in master_patterns)    

df = data_mapping


# %%
# get target data
data_path = "/home/richard/Projects/06_research/hipom_data_mapping/data_preprocess/dataset/1/train_all.csv"
train_df = pd.read_csv(data_path, skipinitialspace=True)
# processing to help with selection later
train_df['thing_property'] = train_df['thing'] + " " + train_df['property']


# %%
condition1 = df['MDM']
condition2 = df['p_MDM']

condition_correct_thing = df['p_thing'] == df['thing']
condition_correct_property = df['p_property'] == df['property']
match = sum(condition1 & condition2)
fn = sum(condition1 & ~condition2)
prediction_mdm_correct = sum(condition_correct_thing & condition_correct_property & condition1)

print("mdm match predicted mdm: ", match)  # 56 - false negative
print("mdm but not predicted mdm: ", fn)  # 56 - false negative
print("total mdm: ", sum(condition1))  # 2113
print("total predicted mdm: ", sum(condition2))  # 6896 - a lot of false positives
print("correct mdm predicted", prediction_mdm_correct)


# %%
# selection
###########################################
# we now have to perform selection
# we restrict to predictions of a class of a ship
# then perform similarity selection with in-distribution data
# the magic is in performing per-class selection, not global
import importlib
import selection
importlib.reload(selection)
selector = selection.Selector(input_df=df, reference_df=train_df)
tp, tn, fp, fn = selector.run_selection(checkpoint_path=checkpoint_path)


# %%
print(tp)
print(tn)
print(fp)
print(fn)
# %%
print("accuracy: ", (tp+tn)/(tp+tn+fp+fn))
print("f1_score: ", (2*tp)/((2*tp) + fp + fn))
print("precision: ", (tp)/(tp+fp))
print("recall: ", (tp)/(tp+fn))
# %%
