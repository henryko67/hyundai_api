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
        self.input_df = input_df
        self.reference_df = reference_df


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
            subset_matrix = cos_sim_matrix[np.ix_(condition_source, condition_target)]
            # we select top k here
            # Get the indices of the top 5 maximum values along axis 1
            top_k = 1
            top_k_indices = np.argsort(subset_matrix, axis=1)[:, -top_k:]  # Get indices of top 5 values
            
            # Get the values of the top 5 maximum scores
            top_k_values = np.take_along_axis(subset_matrix, top_k_indices, axis=1)
            
            # Calculate the average of the top 5 scores along axis 1
            y_scores = np.mean(top_k_values, axis=1)
            max_idx = np.argmax(y_scores)
            max_score = y_scores[max_idx]
            # convert boolean to indices (1,2,3)
            condition_indices = np.where(condition_source)[0]
            max_idx = condition_indices[max_idx]

            return max_idx, max_score


        # prepare reference embed
        train_data = list(generate_input_list(self.reference_df))
        # Define the directory and the pattern
        retriever_train = Retriever(train_data, checkpoint_path)
        retriever_train.make_mean_embedding(batch_size=64)
        train_embed = retriever_train.embeddings

        # take the inputs for df_sub
        test_data = list(generate_input_list(self.input_df))
        retriever_test = Retriever(test_data, checkpoint_path)
        retriever_test.make_mean_embedding(batch_size=64)
        test_embed = retriever_test.embeddings



        # precision_list = []
        # recall_list = []
        tp_accumulate = 0
        tn_accumulate = 0
        fp_accumulate = 0
        fn_accumulate = 0
        THRESHOLD = 0.95
        for ship_idx in self.ships_list:
            print(ship_idx)
            # we select a ship and select only data exhibiting MDM pattern in the predictions
            ship_mask = (self.input_df['ships_idx'] == ship_idx) & (self.input_df['p_MDM'])
            df_ship = self.input_df[ship_mask].reset_index(drop=True)
            # we then try to make a dataframe for each thing_property attribute
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
            test_embed_subset = test_embed[ship_mask]
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


            # # track per ship statistics
            # # subset only selected entries
            # # this subset contains our predicted MDMs
            # df_answer = df_ship.loc[selected_idx_list]

            # relevant = (sum(df_ship['MDM']))
            # retrieved = len(df_answer)
            # true_positive = (sum(df_answer['MDM']))

            # # how many retrieved items are relevant
            # precision = true_positive/retrieved
            # # how many relevant items are retrieved
            # recall = true_positive/relevant

            # precision_list.append(precision)
            # recall_list.append(recall)


            # explanation:
            # we first separated our ship into p_mdm and non p_mdm
            # we only select final in-mdm prediction from p_mdm subset
            # anything that is not selected and from non-p_mdm is predicted not in mdm

            # get our final prediction
            df_subset_predicted_true = df_ship.loc[selected_idx_list]
            # take the set difference between df_ship's index and the given list
            inverse_list = df_ship.index.difference(selected_idx_list).to_list()
            df_subset_predicted_false = df_ship.loc[inverse_list]

            not_p_mdm_mask = (self.input_df['ships_idx'] == ship_idx) & (~self.input_df['p_MDM'])
            df_not_p_mdm = self.input_df[not_p_mdm_mask].reset_index(drop=True)

            # concat
            df_false = pd.concat([df_subset_predicted_false, df_not_p_mdm], axis=0)
            assert(len(df_false) + len(df_subset_predicted_true) == sum(self.input_df['ships_idx'] == ship_idx))




            # true positive -> predicted in mdm, actual in mdm
            # we get all the final predictions that are also found in MDM
            true_positive = sum(df_subset_predicted_true['MDM'])
            # true negative -> predicted not in mdm, and not found in MDM
            # we negate the condition to get those that are not found in MDM
            true_negative = sum(~df_false['MDM'])
            # false positive -> predicted in mdm, not found in mdm
            false_positive = sum(~df_subset_predicted_true['MDM'])
            # false negative -> predicted not in mdm, found in mdm
            false_negative = sum(df_false['MDM'])


            tp_accumulate = tp_accumulate + true_positive
            tn_accumulate = tn_accumulate + true_negative
            fp_accumulate = fp_accumulate + false_positive
            fn_accumulate = fn_accumulate + false_negative

             

        total_sum = (tp_accumulate + tn_accumulate + fp_accumulate + fn_accumulate)
        # ensure that all entries are accounted for
        assert(total_sum == len(self.input_df))
        return tp_accumulate, tn_accumulate, fp_accumulate, fn_accumulate

