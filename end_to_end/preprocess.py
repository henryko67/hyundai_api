# %%
import re
from end_to_end.replacement_dict import desc_replacement_dict, unit_replacement_dict

class Abbreviator:

    def __init__(self, df):
        self.df = df

    def _count_abbreviation_occurrences(self, tag_descriptions, abbreviation):
        """Count the number of occurrences of the abbreviation in the list of machine descriptions."""
        pattern = re.compile(abbreviation)
        count = sum(len(pattern.findall(description)) for description in tag_descriptions)
        return count

    def _replace_abbreviations(self, tag_descriptions, abbreviations):
        """Replace the abbreviations according to the key-pair value provided."""
        replaced_descriptions = []
        for description in tag_descriptions:
            for abbreviation, replacement in abbreviations.items():
                description = re.sub(abbreviation, replacement, description)

            replaced_descriptions.append(description)
        return replaced_descriptions

    def _cleanup_spaces(self, tag_descriptions):
        # Replace all whitespace with a single space
        replaced_descriptions = []
        for description in tag_descriptions:
            description_clean = re.sub(r'\s+', ' ', description)
            replaced_descriptions.append(description_clean)
        return replaced_descriptions

    # remove all dots
    def _cleanup_dots(self, tag_descriptions):
        replaced_descriptions = []
        for description in tag_descriptions:
            description_clean = re.sub(r'\.', '', description)
            replaced_descriptions.append(description_clean)
        return replaced_descriptions


    def run(self):
        df = self.df

        # %%
        # Replace abbreviations
        print("running substitution for descriptions")
        # normalize to uppercase
        # strip leading and trailing whitespace
        df['tag_description'] = df['tag_description'].str.strip()
        df['tag_description'] = df['tag_description'].str.upper()
        # Replace whitespace-only entries with "NOVALUE"
        # note that "N/A" can be read as nan
        # replace whitespace only values as NOVALUE
        df['tag_description']= df['tag_description'].fillna("NOVALUE")
        df['tag_description'] = df['tag_description'].replace(r'^\s*$', 'NOVALUE', regex=True)

        # perform actual substitution
        tag_descriptions = df['tag_description']
        replaced_descriptions = self._replace_abbreviations(tag_descriptions, desc_replacement_dict)
        replaced_descriptions = self._cleanup_spaces(replaced_descriptions)
        replaced_descriptions = self._cleanup_dots(replaced_descriptions)
        df["tag_description"] = replaced_descriptions
        # print("Descriptions after replacement:", replaced_descriptions)

        # %%
        print("running substitutions for units")
        df['unit'] = df['unit'].fillna("NOVALUE")
        df['unit'] = df['unit'].replace(r'^\s*$', 'NOVALUE', regex=True)
        unit_list = df['unit']
        new_unit = self._replace_abbreviations(unit_list, unit_replacement_dict)
        new_unit = self._cleanup_spaces(new_unit)
        df['unit'] = new_unit

        return df

