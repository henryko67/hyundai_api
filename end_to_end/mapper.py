import torch
from torch.utils.data import DataLoader
from transformers import (
    T5TokenizerFast,
    AutoModelForSeq2SeqLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
import os
from tqdm import tqdm
from datasets import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class Mapper():
    tokenizer: T5TokenizerFast
    model: torch.nn.Module
    dataloader: DataLoader

    def __init__(self, checkpoint_path):
        self._create_tokenizer()
        self._load_model(checkpoint_path)


    def _create_tokenizer(self):
        # %%
        # load tokenizer
        self.tokenizer = T5TokenizerFast.from_pretrained("t5-small", return_tensors="pt", clean_up_tokenization_spaces=True)
        # Define additional special tokens
        additional_special_tokens = ["<THING_START>", "<THING_END>", "<PROPERTY_START>", "<PROPERTY_END>", "<NAME>", "<DESC>", "SIG", "UNIT", "DATA_TYPE"]
        # Add the additional special tokens to the tokenizer
        self.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

    def _load_model(self, checkpoint_path: str):
        # load model
        # Define the directory and the pattern
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
        # set model to eval
        self.model = model.eval()
        self.model.cuda()
        # self.model = torch.compile(self.model)





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
            output_list = []
            for _, row in df.iterrows():
                desc = f"<DESC>{row['tag_description']}<DESC>"
                unit = f"<UNIT>{row['unit']}<UNIT>"
                element = {
                    'input' : f"{desc}{unit}"
                    # 'output': f"<THING_START>{row['thing']}<THING_END><PROPERTY_START>{row['property']}<PROPERTY_END>",
                }
                output_list.append(element)

            return output_list

        def _preprocess_function(example):
            input = example['input']
            # target = example['output']
            # text_target sets the corresponding label to inputs
            # there is no need to create a separate 'labels'
            model_inputs = self.tokenizer(
                input,
                # text_target=target, 
                # max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
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

        def _custom_collate_fn(batch):
            # Extract data and targets separately if needed
            inputs = [item['input_ids'] for item in batch]
            attention_masks = [item['attention_mask'] for item in batch]

            # Pad data to the same length
            padded_inputs = pad_sequence(inputs, batch_first=True)
            padded_attention_masks = pad_sequence(attention_masks, batch_first=True)

            return {'input_ids': padded_inputs, 'attention_mask': padded_attention_masks}


        # create dataloader
        self.dataloader = DataLoader(
            datasets, 
            batch_size=batch_size,
            collate_fn=_custom_collate_fn
        )


    def generate(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        MAX_GENERATE_LENGTH = 120

        pred_generations = []
        # pred_labels = []
        # self.model already assigned to device
        # self.model.cuda()

        # introduce early stopping so that it doesn't have to generate max length
        class StopOnEndToken(StoppingCriteria):
            def __init__(self, end_token_id):
                self.end_token_id = end_token_id

            def __call__(self, input_ids, scores, **kwargs):
                # Check if the last token in any sequence is the end token
                batch_stopped = input_ids[:, -1] == self.end_token_id
                # only stop if all have reached end token
                if batch_stopped.all():
                    return True  # Stop generation for the entire batch
                return False

        # Define the end token ID (e.g., the ID for <eos>)
        end_token_id = 32103  # property end token

        # Use the stopping criteria
        stopping_criteria = StoppingCriteriaList([StopOnEndToken(end_token_id)])

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

            # Perform inference
            # disable if running on gpu's without tensor cores
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    outputs = self.model.generate(input_ids,
                                            attention_mask=attention_mask,
                                            max_length=MAX_GENERATE_LENGTH,
                                            # eos_token_id=32103,
                                            # early_stopping=True,
                                            use_cache=True,
                                            stopping_criteria=stopping_criteria)
                
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
