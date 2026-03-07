"""
config.py — All hyperparameters and settings in one place.
"""
import torch


#  Model Architecture 
n_embd    = 384     
num_heads = 6       
n_layer   = 6       
block_size = 256    
dropout   = 0.2     

#  Training 
batch_size     = 64       
max_iters      = 5000     
eval_interval  = 500      
eval_iters     = 200      
learning_rate  = 3e-4     

#  Tokenizer 
vocab_size_limit = 10000  
min_frequency    = 2      


#  Paths 
dataset_path = '../datasets/wikipedia_articles.txt'                 
model_save_path = 'transformer_model.pth'    

#  Device 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#  Reproducibility 
seed = 1330
