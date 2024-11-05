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
import selection

# Create a FastAPI instance
app = FastAPI()

# Define Pydantic models
class Data(BaseModel):
    index: int
    tag_name: str
    equip_type_code: int  # Changed to int
    tag_description: str
    tx_period: int
    tx_type: int
    on_change_yn: bool     # Changed to bool
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

@app.post("/data_mapping/")
async def create_mapping(input: List[Input]):
    # Record the total start time
    total_start_time = time.time()

    # Dictionary to store results for each ships_idx
    results_by_ship = defaultdict(lambda: {"ships_idx": 0, "platform_data_list": [], "unmapped_indices": []})

    # Variable to accumulate inference time across all ships
    total_inference_time = 0
    
    # Iterate over each ship in the input list
    for ship in input:
        ships_idx = ship.ships_idx
        result = results_by_ship[ships_idx]
        result["ships_idx"] = ships_idx

        # Convert ship_data_list into dataframe for the model
        data = [{'tag_description': data.tag_description, 'ships_idx': ships_idx} for data in ship.ship_data_list]
        df = pd.DataFrame(data)

        # Record the inference start time for this ship
        inference_start_time = time.time()

         ##########################################
        # begin inference
       
        # Run inference
        directory = 'checkpoint_epoch40'
        pattern = 'checkpoint-*'
        checkpoint_path = glob.glob(os.path.join(directory, pattern))[0]

        infer = Inference(checkpoint_path)
        infer.prepare_dataloader(df, batch_size=64, max_length=64)
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

        ##########################################
        # begin selection

        # we start to cull predictions from here
        data_master_path = f"data_files/data_model_master_export.csv"
        df_master = pd.read_csv(data_master_path, skipinitialspace=True)
        data_mapping = df
        # Generate patterns    
        df_master['master_pattern'] = df_master['thing'] + " " + df_master['property']    
        # Create a set of unique patterns from master for fast lookup    
        master_patterns = set(df_master['master_pattern'])
        thing_patterns = set(df_master['thing'])

        # check if prediction is in MDM
        data_mapping['p_thing_pattern'] = data_mapping['p_thing'].str.replace(r'\d', '#', regex=True)
        data_mapping['p_property_pattern'] = data_mapping['p_property'].str.replace(r'\d', '#', regex=True)
        data_mapping['p_pattern'] = data_mapping['p_thing_pattern'] + " " + data_mapping['p_property_pattern']
        data_mapping['p_MDM'] = data_mapping['p_pattern'].apply(lambda x: x in master_patterns)    

        df = data_mapping

        # get target data
        data_path = "data_files/train.csv"
        train_df = pd.read_csv(data_path, skipinitialspace=True)
        # processing to help with selection later
        train_df['thing_property'] = train_df['thing'] + " " + train_df['property']

        import selection

        selector = selection.Selector(input_df=df, reference_df=train_df)
        thing_prediction_list, property_prediction_list = selector.run_selection(checkpoint_path=checkpoint_path)


        ##########################################
        # end of inference + selection

        # Record the inference end time for this ship
        inference_end_time = time.time()

        # Calculate inference time for this ship and add to the total inference time
        inference_time_for_ship = inference_end_time - inference_start_time
        total_inference_time += inference_time_for_ship

        # Map predictions back to the input data and assign to result
        for i, data in enumerate(ship.ship_data_list):
            thing = thing_prediction_list[i] if i < len(thing_prediction_list) else None
            property_ = property_prediction_list[i] if i < len(property_prediction_list) else None

            # Check if 'thing' or 'property' is None
            if thing is None or property_ is None:
                result["unmapped_indices"].append(data.index)  # Add index to unmapped
            else:
                result["platform_data_list"].append(
                    ThingProperty(index=data.index, thing=thing, property=property_)
                )
    
    # Record the total end time
    total_end_time = time.time()

    # Calculate total API call time
    total_api_time = total_end_time - total_start_time

    # Convert results_by_ship to a list of results
    final_results = [result for result in results_by_ship.values()]

    return {
        "message": "Data mapped successfully",
        "result": final_results,
        "timing": {
            "total_inference_time": total_inference_time,  # Sum of all ships' inference time
            "total_api_time": total_api_time               # Time for the entire API call
        }
    }

