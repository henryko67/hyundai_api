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
from fastapi import HTTPException
from cache_db import init_cache, get_cache_db_path, get_cache, update_cache, has_cache_key, print_cache_contents, normalize_key

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

@app.post("/data_mapping_cached/")
async def create_mapping(input: List[Input]):
    # Record the total start time
    total_start_time = time.time()

    # Dictionary to store results for each ships_idx
    results_by_ship = defaultdict(lambda: {"ships_idx": 0, "platform_data_list": [], "unmapped_indices": []})

    # Initialize cache for each ship
    for ship in input:
        ships_idx = int(ship.ships_idx)  # Ensure `ships_idx` is an integer
        print(f"Initializing cache for ship {ships_idx}...")
        init_cache(ships_idx)  # Ensure cache table creation

    # Collect data from all ships into a single DataFrame
    df_list = []
    for ship in input:
        ships_idx = int(ship.ships_idx)  # Ensure `ships_idx` is an integer
        data = [
            {
                'index': data.index,
                'tag_description': data.tag_description,
                'unit': data.unit,
                'original_tag_description': data.tag_description,  # Retain original value
                'original_unit': data.unit,  # Retain original value
                'ships_idx': ships_idx
            }
            for data in ship.ship_data_list
        ]
        df = pd.DataFrame(data)
        df_list.append(df)

    df = pd.concat(df_list, axis=0).reset_index(drop=True)

    # Prepare for inference
    to_infer = []

    # Check cache and directly append results for cache hits
    for _, row in df.iterrows():
        try:
            ships_idx = int(row['ships_idx'])  # Ensure `ships_idx` is an integer
            raw_key = (
                row['original_tag_description'] if isinstance(row['original_tag_description'], str) else "",
                row['original_unit'] if isinstance(row['original_unit'], str) else "",
            )
            cache_key = normalize_key(*raw_key)  # Normalize the original key

            if has_cache_key(ships_idx, cache_key):
                thing, property_ = get_cache(ships_idx, cache_key)
                print(f"Retrieving row: {row.to_dict()}")
                if thing is None or property_ is None:
                    print(f"Cache hit with None values for {cache_key} in ship {ships_idx}")
                    results_by_ship[ships_idx]['ships_idx'] = ships_idx
                    results_by_ship[ships_idx]['unmapped_indices'].append(row['index'])
                else:
                    print(f"Cache hit for {cache_key} in ship {ships_idx}: thing={thing}, property={property_}")
                    results_by_ship[ships_idx]['ships_idx'] = ships_idx
                    results_by_ship[ships_idx]['platform_data_list'].append(
                        ThingProperty(index=row['index'], thing=thing, property=property_)
                    )
                    #print("sanity check")
                    #print(ships_idx)
                    #print(results_by_ship[ships_idx])
            else:
                print(f"Cache miss for {cache_key} in ship {ships_idx}")
                to_infer.append(row)
                #print(f"entered row: {row.to_dict()}")
        except Exception as e:
            print(f"[ERROR] Exception while processing row: {row.to_dict()} - {e}")
            raise

    # Perform inference for uncached items
    if to_infer:
        print(f"Performing inference for {len(to_infer)} uncached items...")
        to_infer_df = pd.DataFrame(to_infer)
        to_infer_df = pd.DataFrame(to_infer).reset_index(drop=True)
        inferred_df = run_end_to_end(to_infer_df)

        # Update cache and append inference results
        for _, row in inferred_df.iterrows():
            try:
                print(f"Processing row: {row.to_dict()}")
                ships_idx = int(row['ships_idx'])  # Ensure `ships_idx` is an integer
                raw_key = (
                    row['original_tag_description'] if isinstance(row['original_tag_description'], str) else "",
                    row['original_unit'] if isinstance(row['original_unit'], str) else "",
                )
                cache_key = normalize_key(*raw_key)  # Normalize the original key

                results_by_ship[ships_idx]["ships_idx"] = ships_idx

                update_cache(ships_idx, cache_key, row['p_thing'], row['p_property'])
                print(f"Updated cache for ship {ships_idx}, key {cache_key}")

                if row['p_thing'] is None or row['p_property'] is None:
                    results_by_ship[ships_idx]['unmapped_indices'].append(row['index'])
                else:
                    results_by_ship[ships_idx]['platform_data_list'].append(
                        ThingProperty(index=row['index'], thing=row['p_thing'], property=row['p_property'])
                    )
            except Exception as e:
                print(f"[ERROR] Exception while processing row: {row.to_dict()} - {e}")
                raise

    # Print cache contents for debugging
    #for ship in input:
     #   ships_idx = int(ship.ships_idx)  # Ensure `ships_idx` is an integer
      #  print_cache_contents(ships_idx)

    # Record total API call time
    total_end_time = time.time()
    total_api_time = total_end_time - total_start_time

    # Convert results_by_ship to a list of results
    final_results = [result for result in results_by_ship.values()]

    return {
        "message": "Data mapped successfully",
        "result": final_results,
        "timing": {
            "total_api_time": total_api_time
        }
    }

@app.delete("/flush_cache/")
async def flush_cache():
    """
    Delete all cache files in the cache directory.
    """
    cache_dir = "cache_files"  # Ensure this matches the directory used in get_cache_db_path()

    try:
        # Find all cache files in the directory
        cache_files = glob.glob(os.path.join(cache_dir, "*.db"))

        # Delete each cache file
        for cache_file in cache_files:
            os.remove(cache_file)
            print(f"Deleted cache file: {cache_file}")

        return {
            "message": f"Flushed {len(cache_files)} cache files successfully."
        }
    except Exception as e:
        print(f"Error while flushing cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to flush cache.")
