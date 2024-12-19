import os
import re

import pandas as pd

from end_to_end.post_processing_rules import shaft_rules, ME_rules, GEFlow_rules

class Corrector():

    def __init__(self, df):
        # copy over the existing
        df['edited_p_thing'] = df['p_thing'].copy()
        self.df = df


    def _process_boiler_data(self, boiler_data):
        updated_dataframes = []
        for _, group in boiler_data.groupby('ships_idx'):
            group_copy = group.copy()

            group_copy["tag_description"] = group_copy["tag_description"].fillna('')

            contains_no1_cond = group_copy["tag_description"].str.contains("NO1")
            contains_no2_cond = group_copy["tag_description"].str.contains("NO2")
            contains_aux_cond = group_copy["tag_description"].str.contains("AUXILIARY")
            doesnt_contain_no_cond = ~group_copy["tag_description"].str.contains("NO")
            contains_comp_cond = group_copy["tag_description"].str.contains("COMPOSITE")
            contains_shipboiler_cond = group_copy["edited_p_thing"].str.contains("ShipBoiler")

            group_copy.loc[contains_no1_cond, "edited_p_thing"] = "ShipBoiler1"
            group_copy.loc[contains_no2_cond, "edited_p_thing"] = "ShipBoiler2"
            group_copy.loc[contains_aux_cond&doesnt_contain_no_cond, "edited_p_thing"] = "ShipBoiler1"

            if ((~contains_comp_cond) & (contains_shipboiler_cond)).any():
                max_boiler_number = group_copy.loc[(~contains_comp_cond)&(contains_shipboiler_cond), "edited_p_thing"].str.extract(r'(\d+)$').astype(float).max().fillna(0)[0] + 1
                composite_boiler = f"ShipBoiler{int(max_boiler_number)}"

                if max_boiler_number > 3:
                    max_boiler_number = 3

                group_copy.loc[group_copy["tag_description"].str.contains("COMPOSITE"), "edited_p_thing"] = composite_boiler
            else:
                group_copy.loc[group_copy["tag_description"].str.contains("COMPOSITE"), "edited_p_thing"] = "ShipBoiler1"

            updated_dataframes.append(group_copy)   # Collect updated group

        # Step 2: Concatenate all updated groups
        if (len(updated_dataframes) == 0):
            return boiler_data
        updated_boiler_data = pd.concat(updated_dataframes)
        return updated_boiler_data

    def _check_conditions(self, value, conditions):
        # Check if a value satisfies all conditions
        return all(condition(value) for condition in conditions)

    def _apply_rules(self, description, thing, rules):
    #Processes the description according to the rule table and returns the replacement value if the condition is met, otherwise returns the thing value
        for rule in rules:
            if self._check_conditions(description, rule["conditions"]): #Check that all conditions are met
                return rule["action"]  #Execute the action and return the result
        return thing  #Returns the value of the thing column if it doesn't match any of the rules

    def run_correction(self):
        final_dataframe = self.df.copy()
        # Hwanggawi main function
        #Get partial columns
        TP_df = final_dataframe.loc[:, ['thing', 'property','p_thing','p_property','tag_description','MDM']].copy()

        #Shaft
        SF_df = TP_df[TP_df['thing'].str.contains(('Shaft'), case=False, na=False)]
        SF_df_in_MDM = SF_df[(SF_df['MDM'])]

        #ME
        ME_df = TP_df[TP_df['thing'].str.contains(('ME1Flow'), case=False, na=False)|
                      TP_df['thing'].str.contains(('ME2Flow'), case=False, na=False)|
                      TP_df['thing'].str.contains(('ME3Flow'), case=False, na=False)]
        ME_df_in_MDM = ME_df[(ME_df['MDM'])]

        #GE
        GE_df = TP_df[TP_df['thing'].str.contains(('GE1Flow'), case=False, na=False)|
                      TP_df['thing'].str.contains(('GE2Flow'), case=False, na=False)|
                      TP_df['thing'].str.contains(('GE3Flow'), case=False, na=False)]
        GE_df_in_MDM = GE_df[(GE_df['MDM'])]

        SF_df_in_MDM['standardize_desc'] = SF_df_in_MDM['tag_description'].copy()
        GE_df_in_MDM['standardize_desc'] = GE_df_in_MDM['tag_description'].copy()
        ME_df_in_MDM['standardize_desc'] = ME_df_in_MDM['tag_description'].copy()

        # ShipBoiler class post-processing
        mdm = final_dataframe[final_dataframe["MDM"]].copy()
        boiler_data = mdm[mdm["thing"].str.contains("BOILER")].copy()

        # blr_cond = boiler_data["tag_description"].str.lower().str.contains("BOILER")
        # boiler_cond = boiler_data["tag_description"].str.lower().str.contains("BOILER")
        # boiler_data.shape[0]-(boiler_data[blr_cond].shape[0]+boiler_data[boiler_cond].shape[0])

        # different_cond = boiler_data[~(blr_cond|boiler_cond)].copy()

        # unique_ships_idxs = boiler_data["ships_idx"].unique()

        boiler_data["edited_p_thing"] = boiler_data["p_thing"]
        updated_boiler_data = self._process_boiler_data(boiler_data)
        final_dataframe.loc[updated_boiler_data.index, "edited_p_thing"] = updated_boiler_data["edited_p_thing"]

        result = SF_df_in_MDM.apply(lambda x: self._apply_rules(x['standardize_desc'],x['p_thing'], shaft_rules),axis=1)
        SF_df_in_MDM['edited_p_thing'] = result
        final_dataframe.loc[SF_df_in_MDM.index, "edited_p_thing"] = SF_df_in_MDM['edited_p_thing']

        result = ME_df_in_MDM.apply(lambda x: self._apply_rules(x['standardize_desc'],x['p_thing'], ME_rules),axis=1)
        ME_df_in_MDM['edited_p_thing'] = result
        final_dataframe.loc[ME_df_in_MDM.index, "edited_p_thing"] = ME_df_in_MDM['edited_p_thing']


        result = GE_df_in_MDM.apply(lambda x: self._apply_rules(x['standardize_desc'],x['p_thing'], GEFlow_rules),axis=1)
        GE_df_in_MDM['edited_p_thing'] = result
        final_dataframe.loc[GE_df_in_MDM.index, "edited_p_thing"] = GE_df_in_MDM['edited_p_thing']

        # override p_thing with edited_p_thing
        final_dataframe['p_thing'] = final_dataframe['edited_p_thing'].copy()

        return final_dataframe