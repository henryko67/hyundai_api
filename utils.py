import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import torch.nn.functional as F



class Retriever:
    def __init__(self, input_texts, model_checkpoint):
        # we need to generate the embedding from list of input strings
        self.embeddings = []
        self.inputs = input_texts
        model_checkpoint = model_checkpoint 
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base", return_tensors="pt", clean_up_tokenization_spaces=True)
        # define additional special tokens
        additional_special_tokens = ["<thing_start>", "<thing_end>", "<property_start>", "<property_end>", "<name>", "<desc>", "<sig>", "<unit>", "<data_type>"]
        # add the additional special tokens to the tokenizer
        self.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # device = "cpu"
        model.to(self.device)
        self.model = model.eval()




    def make_mean_embedding(self, batch_size=32):
        all_embeddings = self.embeddings
        input_texts = self.inputs

        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i:i+batch_size]
            # Tokenize the input text
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)


            # Pass the input through the encoder and retrieve the embeddings
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

def cosine_similarity_chunked(batch1, batch2, chunk_size=16):
    batch1_size = batch1.size(0)
    batch2_size = batch2.size(0)
    
    # Prepare an empty tensor to store results
    cos_sim = torch.empty(batch1_size, batch2_size, device=batch1.device)

    # Process batch1 in chunks
    for i in range(0, batch1_size, chunk_size):
        batch1_chunk = batch1[i:i + chunk_size]  # Get chunk of batch1
        
        # Expand batch1 chunk and entire batch2 for comparison
        batch1_chunk_exp = batch1_chunk.unsqueeze(1)  # Shape: (chunk_size, 1, seq_len)
        batch2_exp = batch2.unsqueeze(0)  # Shape: (1, batch2_size, seq_len)
        
        # Compute cosine similarity for the chunk and store it in the final tensor
        cos_sim[i:i + chunk_size] = F.cosine_similarity(batch1_chunk_exp, batch2_exp, dim=-1)
    
    return cos_sim

