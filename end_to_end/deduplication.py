
# %%
import pandas as pd
import os
import glob
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

##################
# global parameters
##################



class BertEmbedder:
    def __init__(self, input_texts, model_checkpoint):
        # we need to generate the embedding from list of input strings
        self.embeddings = []
        self.inputs = input_texts
        model_checkpoint = model_checkpoint 
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt", clean_up_tokenization_spaces=True)

        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu"
        self.model = model.to(self.device)
        self.model = self.model.eval()
        # self.model = torch.compile(self.model)


    def make_embedding(self, batch_size=128):
        all_embeddings = self.embeddings
        input_texts = self.inputs

        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i:i+batch_size]
            # Tokenize the input text
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=120)
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)


            # Pass the input through the encoder and retrieve the embeddings
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    encoder_outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    # get last layer
                    embeddings = encoder_outputs.hidden_states[-1]
                    # get cls token embedding
                    cls_embeddings = embeddings[:, 0, :]  # Shape: (batch_size, hidden_size)
                    all_embeddings.append(cls_embeddings)
        
        # remove the batch list and makes a single large tensor, dim=0 increases row-wise
        all_embeddings = torch.cat(all_embeddings, dim=0)

        self.embeddings = all_embeddings

class T5Embedder:
    def __init__(self, input_texts, model_checkpoint):
        # we need to generate the embedding from list of input strings
        self.embeddings = []
        self.inputs = input_texts
        model_checkpoint = model_checkpoint 
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base", return_tensors="pt", clean_up_tokenization_spaces=True)
        # define additional special tokens
        additional_special_tokens = ["<THING_START>", "<THING_END>", "<PROPERTY_START>", "<PROPERTY_END>", "<NAME>", "<DESC>", "<SIG>", "<UNIT>", "<DATA_TYPE>"]
        # add the additional special tokens to the tokenizer
        self.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu"
        model.to(self.device)
        self.model = model.eval()
        self.model = torch.compile(self.model)




    def make_embedding(self, batch_size=128):
        all_embeddings = self.embeddings
        input_texts = self.inputs

        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i:i+batch_size]
            # Tokenize the input text
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)


            # Pass the input through the encoder and retrieve the embeddings

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    encoder_outputs = self.model.encoder(input_ids, attention_mask=attention_mask)
                    embeddings = encoder_outputs.last_hidden_state

            # Compute the mean pooling of the token embeddings
            # mean_embedding = embeddings.mean(dim=1)
            mean_embedding = (embeddings * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            all_embeddings.append(mean_embedding)
        
        # remove the batch list and makes a single large tensor, dim=0 increases row-wise
        all_embeddings = torch.cat(all_embeddings, dim=0)

        self.embeddings = all_embeddings


def cosine_similarity_chunked(batch1, batch2, chunk_size=1024):
    device = 'cuda'
    batch1_size = batch1.size(0)
    batch2_size = batch2.size(0)
    batch2.to(device)
    
    # Prepare an empty tensor to store results
    cos_sim = torch.empty(batch1_size, batch2_size, device=device)

    # Process batch1 in chunks
    for i in range(0, batch1_size, chunk_size):
        batch1_chunk = batch1[i:i + chunk_size]  # Get chunk of batch1
        
        batch1_chunk.to(device)
        # Expand batch1 chunk and entire batch2 for comparison
        # batch1_chunk_exp = batch1_chunk.unsqueeze(1)  # Shape: (chunk_size, 1, seq_len)
        # batch2_exp = batch2.unsqueeze(0)  # Shape: (1, batch2_size, seq_len)
        batch2_norms = batch2.norm(dim=1, keepdim=True)

        
        # Compute cosine similarity for the chunk and store it in the final tensor
        # cos_sim[i:i + chunk_size] = F.cosine_similarity(batch1_chunk_exp, batch2_exp, dim=-1)

        # Compute cosine similarity by matrix multiplication and normalizing
        sim_chunk = torch.mm(batch1_chunk, batch2.T) / (batch1_chunk.norm(dim=1, keepdim=True) * batch2_norms.T + 1e-8)
        
        # Store the results in the appropriate part of the final tensor
        cos_sim[i:i + chunk_size] = sim_chunk
    
    return cos_sim




###################
# helper functions
class Embedder():
    input_df: pd.DataFrame
    fold: int
    batch_size: int

    def __init__(self, input_df, batch_size):
        self.input_df = input_df
        self.batch_size = batch_size


    def make_embedding(self, checkpoint_path):

        def generate_input_list(df):
            input_list = []
            for _, row in df.iterrows():
                desc = f"<DESC>{row['tag_description']}<DESC>"
                unit = f"<UNIT>{row['unit']}<UNIT>"
                # name = f"<NAME>{row['tag_name']}<NAME>"
                element = f"{desc}{unit}"
                input_list.append(element)
            return input_list

        # prepare reference embed
        train_data = list(generate_input_list(self.input_df))
        # Define the directory and the pattern
        # embedder = T5Embedder(train_data, checkpoint_path)
        embedder = BertEmbedder(train_data, checkpoint_path)
        embedder.make_embedding(batch_size=self.batch_size)
        return embedder.embeddings




# the selection function takes in the full cos_sim_matrix then subsets the
# matrix according to the test_candidates_mask and train_candidates_mask that we
# give it
# it returns the most likely source candidate index and score among the source
# candidate list
# we then map the local idx to the ship-level idx
def selection(cos_sim_matrix, source_mask, target_mask):
    # subset_matrix = cos_sim_matrix[condition_source]
    # except we are subsetting 2D matrix (row, column)
    subset_matrix = cos_sim_matrix[np.ix_(source_mask, target_mask)]
    # we select top-k here
    # Get the indices of the top-k maximum values along axis 1
    top_k = 1
    # returns a potential 2d matrix of which columns have the highest values
    # top_k_indices = np.argsort(subset_matrix, axis=1)[:, -top_k:]  # Get indices of top k values
    # this partial sorts and ensures we care only top_k are correctly sorted
    top_k_indices = np.argpartition(subset_matrix, -top_k, axis=1)[:, -top_k:]
    
    # Get the values of the top 5 maximum scores
    top_k_values = np.take_along_axis(subset_matrix, top_k_indices, axis=1)
    
    # Calculate the average of the top-k scores along axis 1
    y_scores = np.mean(top_k_values, axis=1)
    max_idx = np.argmax(y_scores)
    max_score = y_scores[max_idx]

    # convert boolean to indices
    condition_indices = np.where(source_mask)[0]
    max_idx = condition_indices[max_idx]
    

    return max_idx, max_score



####################
# global level
# obtain the full mdm_list

#####################
# fold level

def run_deduplication(
    test_df,
    train_df,
    batch_size=1024,
    threshold=0.9,
    diagnostic=False
    ):

    # TODO: replace this with a list of values to import
    # too wasteful to just import everything
    data_path = 'end_to_end/data_mapping_mdm.csv'
    full_df = pd.read_csv(data_path, skipinitialspace=True)
    full_df['mapping'] = full_df['thing'] + ' ' + full_df['property']
    full_mdm_mapping_list = sorted(list((set(full_df['mapping']))))

    # set the fold
    # import test data
    df = test_df
    df['p_mapping'] = df['p_thing'] + " " + df['p_property']

    # get target data
    data_path = "end_to_end/train_all.csv"
    train_df = pd.read_csv(data_path, skipinitialspace=True)
    train_df['mapping'] = train_df['thing'] + " " + train_df['property']

    # generate your embeddings
    checkpoint_path = 'end_to_end/models/bert_model'

    # cache embeddings
    file_path = "end_to_end/train_embeds.pt"
    if os.path.exists(file_path):
        # Load the tensor if the file exists
        tensor = torch.load(file_path, weights_only=True)
        print("Loaded tensor")
    else:
        # Create and save the tensor if the file doesn't exist
        print('generate train embeddings')
        train_embedder = Embedder(input_df=train_df, batch_size=batch_size)
        tensor = train_embedder.make_embedding(checkpoint_path)
        torch.save(tensor, file_path, weights_only=True)
        print("Tensor saved to file.")
    
    train_embeds = tensor


    # if we can, we can cache the train embeddings and load directly
    # we can generate the train embeddings once and re-use for every ship

    # generate new embeddings for each ship
    print('generate test embeddings')
    test_embedder = Embedder(input_df=df, batch_size=batch_size)
    global_test_embeds = test_embedder.make_embedding(checkpoint_path)


    # create global_answer array
    # the purpose of this array is to track the classification state at the global
    # level
    global_answer = np.zeros(len(df), dtype=bool)

    #############################
    # ship level
    # we have to split into per-ship analysis
    ships_list = sorted(list(set(df['ships_idx'])))

    for ship_idx in tqdm(ships_list):
        # ship_df = df[df['ships_idx'] == ship_idx]
        # required to map local ship_answer array to global_answer array
        # map_local_index_to_global_index = ship_df.index.to_numpy()

        # we want to subset the ship and only p_mdm values
        ship_mask = df['ships_idx'] == ship_idx
        map_local_index_to_global_index = np.where(ship_mask)[0]
        ship_df = df[ship_mask].reset_index(drop=True)

        # subset the test embeds
        test_embeds = global_test_embeds[map_local_index_to_global_index]

        # generate the cosine sim matrix for the ship level
        cos_sim_matrix = cosine_similarity_chunked(test_embeds, train_embeds, chunk_size=1024).cpu().numpy()

        ##############################
        # selection level
        # The general idea:
        # step 1: keep only pattern generations that belong to mdm list
        # -> this removes totally wrong datasets that mapped to totally wrong things
        # step 2: loop through the mdm list and isolate data in both train and test that
        # belong to the same pattern class
        # -> this is more tricky, because we have non-mdm mapping to correct classes
        # -> so we have to find which candidate is most similar to the training data

        # it is very tricky to keep track of classification across multiple stages so we
        # will use a boolean answer list to map answers back to the global answer list

        # initialize the local answer list
        ship_answer_list = np.ones(len(ship_df), dtype=bool)

        ###########
        # STEP 1A: ensure that the predicted mapping labels are valid
        pattern_match_mask = ship_df['p_mapping'].apply(lambda x: x in full_mdm_mapping_list).to_numpy()
        pattern_match_mask = pattern_match_mask.astype(bool)
        # anything not in the pattern_match_mask are hallucinations
        # this has the same effect as setting any wrong generations as non-mdm
        ship_answer_list[~pattern_match_mask] = False

        # # STEP 1B: subset our de-duplication to use only predicted_mdm labels
        # p_mdm_mask = ship_df['p_mdm']
        # # assign false to any non p_mdm entries
        # ship_answer_list[~p_mdm_mask] = False
        # # modify pattern_match_mask to remove any non p_mdm values
        # pattern_match_mask = pattern_match_mask & p_mdm_mask

        ###########
        # STEP 2
        # we now go through each class found in our generated set

        # we want to identify per-ship mdm classes
        ship_predicted_classes = sorted(set(ship_df['p_mapping'][pattern_match_mask].to_list()))

        # this function performs the selection given a class
        # it takes in the cos_sim_matrix
        # it returns the selection by mutating the answer_list
        # it sets all relevant idxs to False initially, then sets the selected values to True
        def selection_for_class(select_class, cos_sim_matrix, answer_list):

            # create local copy of answer_list
            ship_answer_list = answer_list.copy()
            # sample_df = ship_df[ship_df['p_mapping'] == select_class]


            # we need to set all idx of chosen entries as False in answer_list -> assume wrong by default
            # selected_idx_list = sample_df.index.to_numpy()
            selected_idx_list = np.where(ship_df['p_mapping'] == select_class)[0]

            # basic assumption check

            # generate the masking arrays for both test and train embeddings
            # we select a tuple from each group, and use that as a candidate for selection
            test_candidates_mask = ship_df['p_mapping'] == select_class
            # we make candidates to compare against in the data sharing the same class
            train_candidates_mask = train_df['mapping'] == select_class

            if sum(train_candidates_mask) == 0:
                # it can be the case that the mdm-valid mapping class is not found in training data
                # print("not found in training data", select_class)
                ship_answer_list[selected_idx_list] = False
                return ship_answer_list

            # perform selection
            # max_idx is the id
            max_idx, max_score = selection(cos_sim_matrix, test_candidates_mask, train_candidates_mask)


            # set the duplicate entries to False
            ship_answer_list[selected_idx_list] = False
            # then only set the one unique chosen value as True
            if max_score > threshold:
                ship_answer_list[max_idx] = True

            return ship_answer_list 

        # we choose one mdm class
        for select_class in ship_predicted_classes:
            # this resulted in big improvement
            if (sum(ship_df['p_mapping'] == select_class)) > 0:
                ship_answer_list = selection_for_class(select_class, cos_sim_matrix, ship_answer_list)

        # we want to write back to global_answer
        # first we convert local indices to global indices
        ship_local_indices = np.where(ship_answer_list)[0]
        ship_global_indices = map_local_index_to_global_index[ship_local_indices]
        global_answer[ship_global_indices] = True

    # we set all unselected values to None
    df.loc[~global_answer, 'p_thing'] = None
    df.loc[~global_answer, 'p_property'] = None


    # if diagnostic:
    #     print(80 * '*')

    #     y_true = df['MDM'].to_list()
    #     y_pred = global_answer

    #     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    #     print(f"tp: {tp}")
    #     print(f"tn: {tn}")
    #     print(f"fp: {fp}")
    #     print(f"fn: {fn}")

    #     # compute metrics
    #     accuracy = accuracy_score(y_true, y_pred)
    #     f1 = f1_score(y_true, y_pred)
    #     precision = precision_score(y_true, y_pred)
    #     recall = recall_score(y_true, y_pred)

    #     # print the results
    #     print(f'accuracy: {accuracy:.5f}')
    #     print(f'f1 score: {f1:.5f}')
    #     print(f'Precision: {precision:.5f}')
    #     print(f'Recall: {recall:.5f}')


    return df


