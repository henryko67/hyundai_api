# %%
import pandas as pd
import os
import glob
from end_to_end.mapper import Mapper
from end_to_end.preprocess import Abbreviator
from end_to_end.deduplication import run_deduplication

# global config
BATCH_SIZE = 256


def run_end_to_end(df):
    current_directory = (os.getcwd())
    if ('end_to_end' not in current_directory):
        os.chdir('./end_to_end')

    # pre-process data
    abbreviator = Abbreviator(df)
    df = abbreviator.run()

    # %%
    ##########################################
    # run mapping
    # checkpoint
    # Use glob to find matching paths
    checkpoint_path = 'models/mapping_model'
    mapper = Mapper(checkpoint_path)
    mapper.prepare_dataloader(df, batch_size=BATCH_SIZE, max_length=128)
    thing_prediction_list, property_prediction_list = mapper.generate()

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
    ####################################
    # run de_duplication with thresholding
    data_path = "train_all.csv"
    train_df = pd.read_csv(data_path, skipinitialspace=True)
    train_df['mapping'] = train_df['thing'] + " " + train_df['property']

    df = run_deduplication(
        test_df=df,
        train_df=train_df,
        batch_size=BATCH_SIZE,
        threshold=0.85,
        diagnostic=False)

    return df
