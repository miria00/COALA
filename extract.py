'''
(2)
this file only extracts features from gpt2 model without any type of training

intput: a checkpoint and some data
output: Neg Pos features saved as numpy files; pass into any binary classifier as training data
supports pooling during extraction

Output shape depends on POOL:

none: Shape (batch_size, seq_len, hidden_embedding_size)
mean: Shape (batch_size, hidden_embedding_size)
attn: Shape (batch_size, hidden_embedding_size), weighted by attention scores.

todo: change the names of the MovieReviewDataset and Gpt2ClassificationCollator to generic
CHECK THIS -1 versus +1
'''

from transformers import AutoTokenizer, GPT2Model, set_seed
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import gc
import os
import numpy as np
from LMdata_utils import MovieReviewsDataset_simple, Gpt2ClassificationCollator_simple
from torch.utils.data import DataLoader
import argparse
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM


set_seed(1024)


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--pool", type=str, default="attn")
parser.add_argument("--output_base", type=str, default=None)
args = parser.parse_args()

MODEL_NAME_OR_PATH = args.model_path
DATAPATH = args.data_path
POOL = args.pool


# POOL = 'attn' # none, mean, or attn 
# MODEL_NAME = False

# if MODEL_NAME == True:
#     MODEL_NAME_OR_PATH = "openai-community/gpt2" # HF: gpt2-medium, openai-community/gpt2, etc
# elif MODEL_NAME == False:
#     MODEL_NAME_OR_PATH = "/home/miria/CVXDPO/checkpoints/SFT_openai-community_gpt2_edu" # SFT checkpoint

# # DATAPATH = '/home/miria/jaxopt/GPT2/data/aclImdb_/'
# DATAPATH = "/home/miria/CVXDPO/datasets/edu/"



DATA_POS = 'pos/'
DATA_NEG = 'neg/'
batch_size = 32
max_length = 128 # for all datasets 

if args.output_base:
    OUTPUT_DIR = f'{args.output_base}/extracted_features_{POOL}_NEG_POS_{os.path.basename(os.path.normpath(MODEL_NAME_OR_PATH))}'
else:
    OUTPUT_DIR = f'/home/miria/CVXDPO/extracted_features/extracted_features_{POOL}_NEG_POS_{os.path.basename(os.path.normpath(MODEL_NAME_OR_PATH))}'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

device = torch.device('cuda')
print("using device:", device)
labels_ids = {'neg': 0, 'pos': 1} # assign 0=negative and 1=positive CHECK THIS -1 versus +1 
 
print('Load tokenizer and model from peft...')
#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
#tokenizer.padding_side = "left" # all examples are padded to the same length (take first 60 words ie)
#tokenizer.pad_token = tokenizer.eos_token

#----------------------------------------------------------------------------------------
# below is for loading GPT2 and/or full checkpoints not LORA
# print('Load model...')
# #model = GPT2Model.from_pretrained(MODEL_NAME_OR_PATH) #this needs to be changed to GPT2LMHeadModel, try GPT2Sequence for classification head
# config = AutoConfig.from_pretrained(MODEL_NAME_OR_PATH)
# # AutoModelForCausalLM
# #model = AutoModel.from_pretrained(MODEL_NAME_OR_PATH, config=config)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH, config=config)
from peft import PeftModel, PeftConfig

adapter_path = MODEL_NAME_OR_PATH
peft_config = PeftConfig.from_pretrained(adapter_path)

base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
)

# Load adapter on top of base model and merge for faster inference
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()

# Optional: resize tokenizer if needed
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
tokenizer.padding_side = "left" # all examples are padded to the same length (take first 60 words ie)
tokenizer.pad_token = tokenizer.eos_token


model.resize_token_embeddings(len(tokenizer))
# fix model padding token id
model.config.pad_token_id = model.config.eos_token_id
model = model.to(device)
model.eval()
print('Model on`%s`'%device)


# tokenizer converts raw text into token ID
# collator uses given encoder to convert any text and labels to numbers that can go into GPT
gpt2_classificaiton_collator = Gpt2ClassificationCollator_simple(use_tokenizer=tokenizer, labels_encoder=labels_ids, max_sequence_len=max_length)


class AttentionPooling(torch.nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention_weights = torch.nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        # Ensure we're on the same device as hidden_states
        if self.attention_weights.weight.device != hidden_states.device:
            self.to(hidden_states.device)
            
        # get attention scores
        scores = self.attention_weights(hidden_states)
        scores = scores.squeeze(-1)
        scores = F.softmax(scores, dim=-1)

        # multiply hidden states by attention scores
        context_vector = torch.sum(hidden_states * scores.unsqueeze(-1), dim=1)
        return context_vector

# Load POS and NEG datasets separately first
print(f"POS directory being loaded: {os.path.abspath(DATAPATH + DATA_POS)}")
dataset_inputPOS = MovieReviewsDataset_simple(path=DATAPATH + DATA_POS, use_tokenizer=tokenizer)
print('Created `dataset_inputPOS` with %d examples!'%len(dataset_inputPOS))

print(f"NEG directory being loaded: {os.path.abspath(DATAPATH + DATA_NEG)}")
dataset_inputNEG = MovieReviewsDataset_simple(path=DATAPATH + DATA_NEG, use_tokenizer=tokenizer)
print('Created `dataset_inputNEG` with %d examples!'%len(dataset_inputNEG))

# Create separate dataloaders, but with smaller batch sizes to avoid OOM
batch_size_pos = batch_size
batch_size_neg = batch_size

dataloaderPOS = DataLoader(
    dataset_inputPOS, 
    batch_size=batch_size_pos, 
    shuffle=True, 
    collate_fn=gpt2_classificaiton_collator
)
print('Created `dataloader POS` with %d batches!'%len(dataloaderPOS))

dataloaderNEG = DataLoader(
    dataset_inputNEG, 
    batch_size=batch_size_neg, 
    shuffle=True, 
    collate_fn=gpt2_classificaiton_collator
)
print('Created `dataloader NEG` with %d batches!'%len(dataloaderNEG))

# Create attention pooling layer outside the loop to reuse it
attention_pooling = None

# Process both dataloaders with a single approach
def process_batch(batch, batch_num, prefix, pool_type):
    global attention_pooling
    
    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        # Move batch to GPU
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # Forward pass
        outputs = model(**batch, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        hidden_size = last_hidden_states.size(-1)
        
        # Apply pooling based on type
        if pool_type == 'attn':
            # Create attention pooling if it doesn't exist yet
            if attention_pooling is None:
                attention_pooling = AttentionPooling(hidden_size).to(device=device, dtype=last_hidden_states.dtype)

            # Make sure it's on the right device and dtype
            if attention_pooling.attention_weights.weight.dtype != last_hidden_states.dtype:
                attention_pooling = attention_pooling.to(device=last_hidden_states.device, dtype=last_hidden_states.dtype)

            pooled_output = attention_pooling(last_hidden_states)
            output_np = pooled_output.detach().cpu().float().numpy()
        elif pool_type == 'mean':
            pooled_output = last_hidden_states.mean(dim=1)
            output_np = pooled_output.detach().cpu().float().numpy()
        else:
            output_np = last_hidden_states.detach().cpu().float().numpy()
            
        # Save to file
        np.save(os.path.join(OUTPUT_DIR, f'{prefix}last_hidden_states_{batch_num}.npy'), output_np)
    
    # Force garbage collection to reduce memory usage
    if batch_num % 10 == 0:
        torch.cuda.empty_cache()
        gc.collect()

# Process loops with the fixed function
print("\nProcessing POS samples:")
for batch_num, batch in enumerate(tqdm(dataloaderPOS, desc="POS Feature Extraction")):
    process_batch(batch, batch_num+1, "POS", POOL)

print("\nProcessing NEG samples:")
for batch_num, batch in enumerate(tqdm(dataloaderNEG, desc="NEG Feature Extraction")):
    process_batch(batch, batch_num+1, "NEG", POOL)

# Print final statistics
print("\nExtraction complete!")
# print(f"Total POS batches processed: {total_batches_pos}")
# print(f"Total NEG batches processed: {total_batches_neg}")

# if POOL == 'attn' or POOL == 'mean':
#     print("Done! Shape of output is batch_size x hidden_embedding_size: ", 
#           batch_size, "x", last_hidden_states.shape[-1])
#     print(f"Output is_{POOL}_Pooled, numpy files saved and ready for classification!")
# else:
#     print("Done! Shape of output is batch_size x seq_len x hidden_embedding_size: ", 
#           batch_size, "x", max_length, " x", last_hidden_states.shape[-1])
#     print(f"Output is_{MODEL_NAME_OR_PATH}, numpy files saved and ready for classification!")

print("Output directory is: ", OUTPUT_DIR)

''''
example output: 
number of files created should be number of batches in dataloader:  92
Done! shape of output should be batch_size x hidden_embedding_size:  150 x 768
Output is_attn_Pooled, numpy files saved and ready for classification with your classifier of choice!
Output directory is:  /home/miria/CVXDPO/extracted_features_attn_NEG_POS_SFT_openai-community_gpt2_ultra

Number of files created should be number of batches in dataloader:  159
Done! shape of output should be batch_size x hidden_embedding_size:  150 x 768
Output is_attn_Pooled, numpy files saved and ready for classification with your classifier of choice!
Output directory is:  /home/miria/CVXDPO/extracted_features_attn_NEG_POS_SFT_openai-community_gpt2_edu
'''