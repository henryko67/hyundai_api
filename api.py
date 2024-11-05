from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as numpy
import os
import glob
from inference import Inference
from selection import Selector
from collections import defaultdict
print(os.getcwd())
os.chdir('/home/henryko67/Projects/selection')

# Create a FastAPI instance
app = FastAPI()

# Define Pydantic models
class Data(BaseModel):
    index: int
    tag_name: str
    equip_type_code: str
    tag_description: str
    tx_period: int
    tx_type: int
    on_change_yn: str
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
    thing: str
    property: str

class Result(BaseModel):
    ships_idx: int
    platform_data_list: List[ThingProperty]
    unmapped_indices: List[int]

@app.post("/data_mapping/")
async def create_mapping(input: List[Input]):
    # Dictionary to store results for each ships_idx
    results_by_ship = defaultdict(lambda: {"ships_idx": 0, "platform_data_list": [], "unmapped_indices": []})
    
    # Iterate over each ship in the input list
    for ship in input:
        ships_idx = ship.ships_idx
        # Initialize the result structure for this ships_idx
        result = results_by_ship[ships_idx]
        result["ships_idx"] = ships_idx

        # Convert ship_data_list into dataframe for the model
        data = [{'tag_description': data.tag_description, 'ships_idx': ships_idx} for data in ship.ship_data_list]
        df = pd.DataFrame(data)

        # Run inference
        directory = 'checkpoint_epoch40'
        pattern = 'checkpoint-*'
        checkpoint_path = glob.glob(os.path.join(directory, pattern))[0]

        infer = Inference(checkpoint_path)
        infer.prepare_dataloader(df, batch_size=256, max_length=64)
        thing_prediction_list, property_prediction_list = infer.generate()

        # Map predictions back to the input data and assign to result
        for i, data in enumerate(ship.ship_data_list):
            thing = thing_prediction_list[i] if i < len(thing_prediction_list) else "Unknown"
            property_ = property_prediction_list[i] if i < len(property_prediction_list) else "Unknown"
            result["platform_data_list"].append(
                ThingProperty(index=data.index, thing=thing, property=property_)
            )
    
    # Convert results_by_ship to a list of results
    final_results = [result for result in results_by_ship.values()]

    return {
        "message": "Data mapped successfully",
        "result": final_results
    }
