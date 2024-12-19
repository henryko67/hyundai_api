import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
import glob
from run_end_to_end import run_end_to_end
from collections import defaultdict
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import selection

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
    inference_start_time = time.time()

    global_train_embed = None
    
    # cache here
    #####
    # code here
    #####
    df_list = []
    # create combined dataframe such that ships_idx is intruded into the ship_data_list
    for ship in input:
        ships_idx = ship.ships_idx

        # i need the data to be a single df
        data = [{
            'index': data.index,
            'tag_description': data.tag_description,
            'unit': data.unit,
            'ships_idx': ships_idx, 
            } 
            for data in ship.ship_data_list]
        df = pd.DataFrame(data)

        df_list.append(df)

    df = pd.concat(df_list, axis=0)
    df = df.reset_index()
 

        
    ###############################
    # start of inference + process
    df = run_end_to_end(df)

    # Record the inference end time for this ship
    inference_end_time = time.time()

    # Calculate inference time for this ship and add to the total inference time
    total_inference_time = inference_end_time - inference_start_time 


    ###############################
    # extract the result back to each ship

    ##########################################
    # end of inference + selection

    for ship in input:
        ships_idx = ship.ships_idx
        result = results_by_ship[ships_idx]
        result['ships_idx'] = ships_idx
        
        # subset df based on ship
        ship_mask = (df['ships_idx'] == ships_idx)
        ship_df = df[ship_mask].reset_index(drop=True)

        for i, row in ship_df.iterrows():
            index = row['index']
            thing = row['p_thing']
            property = row['p_property']

            if (thing is None) or (property is None):
                result['unmapped_indices'].append(index)
            else:
                result["platform_data_list"].append(
                    ThingProperty(index=index, thing=thing, property=property)
                )



    # record total time
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

