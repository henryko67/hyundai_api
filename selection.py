import pandas as pd
import numpy as np
from typing import List
from tqdm import tqdm
from utils import Retriever, cosine_similarity_chunked

class Selector():
    input_df: pd.DataFrame
    reference_df: pd.DataFrame
    ships_list: List[int]

    def __init__(self, input_df, reference_df):
        self.ships_list = sorted(list(set(input_df['ships_idx'])))
        self.input_df = input_df.copy()
        self.reference_df = reference_df.copy()


    def run_selection(self, checkpoint_path):

        def generate_input_list(df):
            input_list = []
            for _, row in df.iterrows():
                desc = f"<DESC>{row['tag_description']}<DESC>"
                element = f"{desc}"
                input_list.append(element)
            return input_list

        # given a dataframe, return a single idx of the entry has the highest match with
        # the embedding
        def selection(cos_sim_matrix, condition_source, condition_target):
            # subset_matrix = cos_sim_matrix[condition_source]
            # except we are subsetting 2D matrix (row, column)
            # we are able to subset the cos_sim_matrix by passing masks
            # condition_source: mask for input subsets
            # condition_target: mask for output subsets
            subset_matrix = cos_sim_matrix[np.ix_(condition_source, condition_target)]
            # we select top k here
            # Get the indices of the top 5 maximum values along axis 1
            top_k = 1
            top_k_indices = np.argsort(subset_matrix, axis=1)[:, -top_k:]  # Get indices of top 5 values
            
            # Get the values of the top 5 maximum scores
            top_k_values = np.take_along_axis(subset_matrix, top_k_indices, axis=1)
            
            # Calculate the average of the top 5 scores along axis 1
            y_scores = np.mean(top_k_values, axis=1)
            local_max_idx = np.argmax(y_scores)
            max_score = y_scores[local_max_idx]
            # convert boolean to indices (1,2,3)
            condition_indices = np.where(condition_source)[0]
            # among global indices, we select the max_idx
            # this acts as a map from subset to the global embedding index list
            global_max_idx = condition_indices[local_max_idx]

            return global_max_idx, max_score


        # print('create embeddings for train_data')
        # prepare reference embed
        train_data = list(generate_input_list(self.reference_df))
        # Define the directory and the pattern
        retriever_train = Retriever(train_data, checkpoint_path)
        retriever_train.make_mean_embedding(batch_size=64)
        train_embed = retriever_train.embeddings

        # take the inputs for df_sub
        # print('create embeddings for train_data')
        test_data = list(generate_input_list(self.input_df))
        retriever_test = Retriever(test_data, checkpoint_path)
        retriever_test.make_mean_embedding(batch_size=64)
        test_embed = retriever_test.embeddings


        # after we create the embeddings, we deal with whole embeddings.
        # we subset these embeddings by applying masks to them


        THRESHOLD = 0.0
        # print(self.ships_list)
        for ship_idx in self.ships_list:
            # print("ship: ", ship_idx)
            # we select a ship and select only data exhibiting MDM pattern in the predictions
            ship_mdm_mask = (self.input_df['ships_idx'] == ship_idx) & (self.input_df['p_MDM'])
            df_ship = self.input_df[ship_mdm_mask]
            # we save the original df index so that we can map the df_ship entries back to the input_df index
            map_back_to_global_id = df_ship.index.to_list()
            # create a copy so that we do not write to view
            df_ship = df_ship.reset_index(drop=True)
            # we then try to make masks for each thing_property attribute
            df_ship['thing_property'] = df_ship['p_thing'] + " " + df_ship['p_property']
            unique_patterns = list(set(df_ship['thing_property']))
            condition_list = []
            for pattern in unique_patterns:
                condition_source = (df_ship['thing_property'] == pattern)
                condition_target = (self.reference_df['thing_property'] == pattern)
                item = {'condition_source': condition_source,
                        'condition_target': condition_target}
                condition_list.append(item)

            # subset part of self.input_df that belongs to the ship 
            test_embed_subset = test_embed[ship_mdm_mask]
            # print('compute cosine')
            cos_sim_matrix = cosine_similarity_chunked(test_embed_subset, train_embed, chunk_size=8).cpu().numpy()


            # for each sub_df, we have to select the best candidate
            # we will do this by finding which desc input has the highest similarity with train data
            selected_idx_list = []
            similarity_score = []
            for item in tqdm(condition_list):
                condition_source = item['condition_source']
                condition_target = item['condition_target']
                # if there is no equivalent data in target, we skip
                if sum(condition_target) == 0:
                    # print("skipped")
                    pass
                # if there is equivalent data in target, we perform selection among source
                # by top-k highest similarity with targets
                else:
                    max_idx, max_score = selection(
                        cos_sim_matrix, condition_source, condition_target 
                    )
                    # implement thresholding
                    if max_score > THRESHOLD:
                        selected_idx_list.append(max_idx)
                        similarity_score.append(max_score)



            # print('selected ids', selected_idx_list)
            # explanation:
            # we first separated our ship into p_mdm and non p_mdm
            # we only select final in-mdm prediction from p_mdm subset
            # anything that is not selected and from non-p_mdm is predicted not in mdm

            # get our final prediction
            # df_subset_predicted_true = df_ship.loc[selected_idx_list]
            # take the set difference between df_ship's index and the given list
            selected_list_global = [map_back_to_global_id[idx] for idx in selected_idx_list]
            inverse_list = self.input_df.index.difference(selected_list_global).to_list()
            # df_subset_predicted_false = df_ship.loc[inverse_list]
            self.input_df.loc[inverse_list, 'p_thing'] = None  
            self.input_df.loc[inverse_list, 'p_property'] = None  

        thing_prediction_list = self.input_df['p_thing'].to_list()
        property_prediction_list = self.input_df['p_property'].to_list()

        # print(len(df_ship)) 
        # print(len(self.input_df))
        assert(len(thing_prediction_list) == len(property_prediction_list))
        assert(len(thing_prediction_list) == len(self.input_df))
        return thing_prediction_list, property_prediction_list

