# %%
import torch
from transformers import (
    T5TokenizerFast,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments
)
import evaluate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
import numpy as np


from datasets import load_from_disk
import os

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

torch.set_float32_matmul_precision('medium')

# %%
# first load the full dataset
# create function to generate dataset
# raw data
data_path = f"/home/richard/Projects/06_research/hipom_data_mapping/data_import/raw_data.csv"
df = pd.read_csv(data_path)
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
data_mapping['thing_MDM'] = data_mapping['thing_pattern'].apply(lambda x: x in thing_patterns)    
df = data_mapping

# outputs a list of dictionaries
def process_df(df):
    output_list = []
    for _, row in df.iterrows():
        # name = f"<NAME>{row['tag_name']}<NAME>"
        desc = f"<DESC>{row['tag_description']}<DESC>"
        # signal = f"<SIG>{row['signal_type']}<SIG>"
        # unit = f"<UNIT>{row['unit']}<UNIT>"
        # data_type = f"<DATA_TYPE>{row['data_type']}<DATA_TYPE>"
        element = {
            'input' : f"{desc}",
            'output': f"<THING_START>{row['thing']}<THING_END><PROPERTY_START>{row['property']}<PROPERTY_END>",
        }
        output_list.append(element)

    return output_list


# %%
def create_split_dataset():
    # train 
    data_path = "/home/richard/Projects/06_research/hipom_data_mapping/data_preprocess/dataset/1/train.csv"
    train_df = pd.read_csv(data_path, skipinitialspace=True)

    # valid
    data_path = "/home/richard/Projects/06_research/hipom_data_mapping/data_preprocess/dataset/1/valid.csv"
    validation_df = pd.read_csv(data_path, skipinitialspace=True)

    combined_data = DatasetDict({
        'train': Dataset.from_list(process_df(train_df)),
        'validation' : Dataset.from_list(process_df(validation_df)),
    })
    return combined_data

# %%
# load training


# %%
# import data and load dataset
# Path to saved combined_dataset

def main():
    save_path = 'checkpoint_epoch40'
    split_datasets = create_split_dataset()

    # prepare tokenizer

    model_checkpoint = "t5-base"
    tokenizer = T5TokenizerFast.from_pretrained(model_checkpoint, return_tensors="pt", clean_up_tokenization_spaces=True)
    # Define additional special tokens
    additional_special_tokens = ["<THING_START>", "<THING_END>", "<PROPERTY_START>", "<PROPERTY_END>", "<NAME>", "<DESC>", "<SIG>", "<UNIT>", "<DATA_TYPE>"]
    # Add the additional special tokens to the tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

    max_length = 120

    # given a dataset entry, run it through the tokenizer
    def preprocess_function(example):
        input = example['input']
        target = example['output']
        # text_target sets the corresponding label to inputs
        # there is no need to create a separate 'labels'
        model_inputs = tokenizer(
            input,
            text_target=target, 
            max_length=max_length,
            truncation=True,
            padding=True
        )
        return model_inputs

    # map maps function to each "row" in the dataset
    # aka the data in the immediate nesting
    tokenized_datasets = split_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=8,
        remove_columns=split_datasets["train"].column_names,
    )


    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    # important! after extending tokens vocab
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric = evaluate.load("sacrebleu")


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, 
                                            skip_special_tokens=False)

        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels,
                                                skip_special_tokens=False)

        # Remove <PAD> tokens from decoded predictions and labels
        decoded_preds = [pred.replace(tokenizer.pad_token, '').strip() for pred in decoded_preds]
        decoded_labels = [[label.replace(tokenizer.pad_token, '').strip()] for label in decoded_labels]

        # Some simple post-processing
        # decoded_preds = [pred.strip() for pred in decoded_preds]
        # decoded_labels = [[label.strip()] for label in decoded_labels]
        # print(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}


    # Generation Config
    # from transformers import GenerationConfig
    gen_config = model.generation_config
    gen_config.max_length = 64

    # compile
    # model = torch.compile(model, backend="inductor", dynamic=True)


    # Trainer

    args = Seq2SeqTrainingArguments(
        f"{save_path}",
        eval_strategy="epoch",
        logging_dir="tensorboard-log",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=1e-4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        auto_find_batch_size=False,
        ddp_find_unused_parameters=False,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=40,
        predict_with_generate=True,
        bf16=True,
        push_to_hub=False,
        generation_config=gen_config,
        remove_unused_columns=False,
    )


    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # uncomment to load training from checkpoint
    # checkpoint_path = 'default_40_1/checkpoint-5600'
    # trainer.train(resume_from_checkpoint=checkpoint_path)

    trainer.train()

# Using the special variable 
if __name__=="__main__":
    main()
