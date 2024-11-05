import torch
from torch.utils.data import DataLoader
from transformers import (
    T5TokenizerFast,
    AutoModelForSeq2SeqLM,
)
import glob
import os
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
import numpy as np

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class Inference():
    tokenizer: T5TokenizerFast
    model: torch.nn.Module
    dataloader: DataLoader

    def __init__(self, checkpoint_path):
        self._create_tokenizer()
        self._load_model(checkpoint_path)


    def _create_tokenizer(self):
        # %%
        # load tokenizer
        self.tokenizer = T5TokenizerFast.from_pretrained("t5-base", return_tensors="pt", clean_up_tokenization_spaces=True)
        # Define additional special tokens
        additional_special_tokens = ["<THING_START>", "<THING_END>", "<PROPERTY_START>", "<PROPERTY_END>", "<NAME>", "<DESC>", "SIG", "UNIT", "DATA_TYPE"]
        # Add the additional special tokens to the tokenizer
        self.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

    def _load_model(self, checkpoint_path: str):
        # load model
        # Define the directory and the pattern
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
        model = torch.compile(model)
        # set model to eval
        self.model = model.eval()




    def prepare_dataloader(self, input_df, batch_size, max_length):
        """
        *arguments*
        - input_df: input dataframe containing fields 'tag_description', 'thing', 'property'
        - batch_size: the batch size of dataloader output
        - max_length: length of tokenizer output
        """
        print("preparing dataloader")
        # convert each dataframe row into a dictionary
        # outputs a list of dictionaries
        def _process_df(df):
            output_list = [{
                    'input': f"<DESC>{row['tag_description']}<DESC>",
                    # 'output': f"<THING_START>{row['thing']}<THING_END><PROPERTY_START>{row['property']}<PROPERTY_END>",
            } for _, row in df.iterrows()]

            return output_list

        def _preprocess_function(example):
            input = example['input']
            # target = example['output']
            # text_target sets the corresponding label to inputs
            # there is no need to create a separate 'labels'
            model_inputs = self.tokenizer(
                input,
                # text_target=target, 
                max_length=max_length,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
            )
            return model_inputs

        test_dataset = Dataset.from_list(_process_df(input_df))


        # map maps function to each "row" in the dataset
        # aka the data in the immediate nesting
        datasets = test_dataset.map(
            _preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=test_dataset.column_names,
        )
        # datasets = _preprocess_function(test_dataset)
        datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        print(type(datasets))

        # create dataloader
        self.dataloader = DataLoader(datasets, batch_size=batch_size)


    def generate(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MAX_GENERATE_LENGTH = 128

        pred_generations = []
        pred_labels = []

        print("start generation")
        for batch in tqdm(self.dataloader):
            # Inference in batches
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            # save labels too
            # pred_labels.extend(batch['labels'])
            

            # Move to GPU if available
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            self.model.to(device)

            # Perform inference
            with torch.no_grad():
                outputs = self.model.generate(input_ids,
                                        attention_mask=attention_mask,
                                        max_length=MAX_GENERATE_LENGTH)
                
                # Decode the output and print the results
                pred_generations.extend(outputs.to("cpu"))



        # %%
        # extract sequence and decode
        def extract_seq(tokens, start_value, end_value):
            if start_value not in tokens or end_value not in tokens:
                return None  # Or handle this case according to your requirements
            start_id = np.where(tokens == start_value)[0][0]
            end_id = np.where(tokens == end_value)[0][0]

            return tokens[start_id+1:end_id]


        def process_tensor_output(tokens):
            thing_seq = extract_seq(tokens, 32100, 32101) # 32100 = <THING_START>, 32101 = <THING_END>
            property_seq = extract_seq(tokens, 32102, 32103) # 32102 = <PROPERTY_START>, 32103 = <PROPERTY_END>
            p_thing = None
            p_property = None
            if (thing_seq is not None):
                p_thing =  self.tokenizer.decode(thing_seq, skip_special_tokens=False)
            if (property_seq is not None):
                p_property =  self.tokenizer.decode(property_seq, skip_special_tokens=False)
            return p_thing, p_property

        # decode prediction labels
        def decode_preds(tokens_list):
            thing_prediction_list = []
            property_prediction_list = []
            for tokens in tokens_list:
                p_thing, p_property = process_tensor_output(tokens)
                thing_prediction_list.append(p_thing)
                property_prediction_list.append(p_property)
            return thing_prediction_list, property_prediction_list 

        thing_prediction_list, property_prediction_list = decode_preds(pred_generations)
        return thing_prediction_list, property_prediction_list

