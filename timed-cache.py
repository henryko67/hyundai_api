import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
import glob
from inference import Inference
from selection import Selector
from collections import defaultdict
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import selection
from cache_db import init_cache, get_cache, update_cache, has_cache_key

# Create a FastAPI instance
app = FastAPI()

# Allow requests from any origin when developing with ngrok
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the cache database
init_cache()

# Define Pydantic models
class Data(BaseModel):
    index: int
    tag_name: str
    equip_type_code: int
    tag_description: str
    tx_period: int
    tx_type: int
    on_change_yn: bool
    scaling_const: int
    signal_type: str
    min: int
    max: int
    unit: str
    data_type: int

class Input(BaseModel):
    ships_idx: int
    ship_data_list: List[Data]

class ThingProperty(BaseModel):
    index: int
    thing: Optional[str] = None
    property: Optional[str] = None

class Result(BaseModel):
    ships_idx: int
    platform_data_list: List[ThingProperty]
    unmapped_indices: List[int]

@app.post("/data_mapping_cached/")
async def create_mapping_cached(input: List[Input]):
    # Record the total start time
    total_start_time = time.time()

    # Dictionary to store results for each ships_idx
    results_by_ship = defaultdict(lambda: {"ships_idx": 0, "platform_data_list": [], "unmapped_indices": []})
    total_inference_time = 0
    global_train_embed = None

    # Iterate over each ship in the input list
    for ship in input:
        ships_idx = ship.ships_idx
        result = results_by_ship[ships_idx]
        result["ships_idx"] = ships_idx

        # Prepare input data for cache check
        data = [{'index': data.index, 'tag_description': data.tag_description, 'unit': data.unit, 'ships_idx': ships_idx}
                for data in ship.ship_data_list]
        df = pd.DataFrame(data)

        # Check cache for all (tag_description, unit) keys
        cached_results = {}
        to_infer = []

        for _, row in df.iterrows():
            cache_key = (row['tag_description'], row['unit'])  # Use composite key
            if has_cache_key(cache_key):  # Check if the key exists in the cache
                thing, property_ = get_cache(cache_key)  # Fetch the cached values
                if thing is None or property_ is None:
                    # Cache hit with None values, add to unmapped
                    result["unmapped_indices"].append(row['index'])
                else:
                    # Valid cache hit, add to cached results
                    cached_results[row['index']] = (thing, property_)
            else:
                # True cache miss, add to inference list
                to_infer.append(row)

        # If we have items to infer, run them in bulk
        if to_infer:
            print("Running inference and selection for uncached items...")
            print(to_infer)

            to_infer_df = pd.DataFrame(to_infer).reset_index(drop=True)
            directory = 'checkpoint_epoch40'
            checkpoint_path = glob.glob(os.path.join(directory, 'checkpoint-*'))[0]

            # Run inference on the batch
            infer = Inference(checkpoint_path)
            infer.prepare_dataloader(to_infer_df, batch_size=64, max_length=64)
            infer_thing_list, infer_property_list = infer.generate()

            # Add inference results to DataFrame for processing
            to_infer_df['p_thing'] = infer_thing_list
            to_infer_df['p_property'] = infer_property_list

            # Perform MDM logic
            data_master_path = "data_files/data_model_master_export.csv"
            df_master = pd.read_csv(data_master_path, skipinitialspace=True)
            df_master['master_pattern'] = df_master['thing'] + " " + df_master['property']
            master_patterns = set(df_master['master_pattern'])

            # Handle None values for string operations
            to_infer_df['p_thing'] = to_infer_df['p_thing'].fillna('').astype(str)
            to_infer_df['p_property'] = to_infer_df['p_property'].fillna('').astype(str)

            to_infer_df['p_thing_pattern'] = to_infer_df['p_thing'].str.replace(r'\d', '#', regex=True)
            to_infer_df['p_property_pattern'] = to_infer_df['p_property'].str.replace(r'\d', '#', regex=True)
            to_infer_df['p_pattern'] = to_infer_df['p_thing_pattern'] + " " + to_infer_df['p_property_pattern']
            to_infer_df['p_MDM'] = to_infer_df['p_pattern'].apply(lambda x: x in master_patterns)

            # Restore None values for unmapped indices
            to_infer_df.loc[to_infer_df['p_thing'] == '', 'p_thing'] = None
            to_infer_df.loc[to_infer_df['p_property'] == '', 'p_property'] = None

            # Perform selection
            data_path = "data_files/train.csv"
            train_df = pd.read_csv(data_path, skipinitialspace=True)
            train_df['thing_property'] = train_df['thing'] + " " + train_df['property']

            selector = Selector(input_df=to_infer_df, reference_df=train_df, global_train_embed=global_train_embed)
            selected_thing_list, selected_property_list = selector.run_selection(checkpoint_path=checkpoint_path)
            global_train_embed = selector.global_train_embed

            # Update cache with selected results
            for i, row in to_infer_df.iterrows():
                cache_key = (row['tag_description'], row['unit'])  # Use composite key
                index = row['index']
                thing = selected_thing_list[i]
                property_ = selected_property_list[i]
                update_cache(cache_key, thing, property_)
                cached_results[index] = (thing, property_)

                # Handle unmapped indices after inference
                if thing is None or property_ is None:
                    result["unmapped_indices"].append(index)

        else:
            print("All requests were cache hits. Skipping inference and selection.")

        # Combine cached and inferred results
        thing_prediction_list = []
        property_prediction_list = []

        for _, row in df.iterrows():
            index = row['index']
            thing, property_ = cached_results.get(index, (None, None))
            thing_prediction_list.append(thing)
            property_prediction_list.append(property_)

        # Add predictions to the DataFrame
        df['p_thing'] = thing_prediction_list
        df['p_property'] = property_prediction_list

        # Record the inference end time for this ship
        inference_end_time = time.time()
        inference_time_for_ship = inference_end_time - total_start_time
        total_inference_time += inference_time_for_ship

        # Map predictions back to the input data and assign to result
        for i, data in enumerate(ship.ship_data_list):
            thing = thing_prediction_list[i] if i < len(thing_prediction_list) else None
            property_ = property_prediction_list[i] if i < len(property_prediction_list) else None

            if thing is None or property_ is None:
                # Append only if not already added during cache check or inference
                if data.index not in result["unmapped_indices"]:
                    result["unmapped_indices"].append(data.index)
            else:
                result["platform_data_list"].append(
                    ThingProperty(index=data.index, thing=thing, property=property_)
                )

    # Record the total end time
    total_end_time = time.time()
    total_api_time = total_end_time - total_start_time

    final_results = [result for result in results_by_ship.values()]

    return {
        "message": "Data mapped successfully",
        "result": final_results,
        "timing": {
            "total_inference_time": total_inference_time,
            "total_api_time": total_api_time
        }
    }
