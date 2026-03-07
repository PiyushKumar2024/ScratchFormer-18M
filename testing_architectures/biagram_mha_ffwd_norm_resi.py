"""Will struggle with vanishing gradients"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer
import os

model_path = 'transformer_model.pth'

# Hyperparameters
batch_size = 32
block_size = 32
max_iters = 20000
eval_interval = 1000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd=32 #will define the feature of every token
num_heads=4 #number of attention heads
dropout=0.2 #dropout rate for regularisation(andrej style)

#for reproducibility
torch.manual_seed(1330)

#Data Loading & Preprocessing
# NOTE: If you run this outside of Colab, make sure to update this file path
filepath = 'dataset.txt'

with open(filepath, 'r', encoding='utf-8-sig') as f:
    textData = f.read()

# Removing the unnecessary part and keeping only the main content
start_text = '*** START OF THE PROJECT GUTENBERG EBOOK'
end_text = '*** END OF THE PROJECT GUTENBERG EBOOK'
start_ind = textData.find(start_text)
end_ind = textData.find(end_text)

if start_ind != -1 and end_ind != -1:
    start = textData.find('\n', start_ind) + 1
    textData = textData[start:end_ind].strip()

print(f"Dataset length: {len(textData)} characters")

#Tokenizer Setup (Custom BPE) from hugging face 
tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(
    [textData], #data
    vocab_size=5000, #max vocab/distinct tokens
    min_frequency=2, #atleast 2 freq
    special_tokens=["<|endoftext|>"]
)
vocab_size = tokenizer.get_vocab_size()
print(f"Vocabulary size: {vocab_size}")

def encode(s):
    return tokenizer.encode(s).ids

def decode(l):
    return tokenizer.decode(l)

#Data Splitting & Batching
# Convert the entire dataset into a tensor
data = torch.tensor(encode(textData), dtype=torch.long)

# Splitting into train and val (85/15)
n = int(len(data) * 0.85)
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # Pick the correct dataset
    data_source = train_data if split == 'train' else val_data
    
    # Pick random starting indices
    randStartInd = torch.randint(0, len(data_source) - block_size, (batch_size,))
    
    # Extract block_size chunks and stack them
    x = torch.stack([data_source[i:i+block_size] for i in randStartInd])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in randStartInd])
    
    # move twnsor to cuda if avail
    x, y = x.to(device), y.to(device)
    return x, y

#Loss Estimation(after every certain iter)
@torch.no_grad() # Tells PyTorch not to waste memory on gradients here
def estimate_loss():
    out = {}
    model.eval() #putting model on eval(baad mein tranformer mein kam ayega)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train() # Put model back in training mode
    return out

class Head(nn.Module):
    """ one head of self-attention """
    #channel is here equal to n_embd

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) #the key layer
        self.query = nn.Linear(n_embd, head_size, bias=False) #the query lyer
        self.value = nn.Linear(n_embd, head_size, bias=False) #the value layer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #for masking of the future
        self.dropout=nn.Dropout(dropout) #dropout after softmax(andrej style)

    def forward(self, x):
        B,T,C = x.shape
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)
        wei=q@k.transpose(-2,-1)*(k.shape[-1]**-0.5) #(b,t,t) #k.shape[-1] will have head_size
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) #(b,t,t)
        wei=F.softmax(wei,dim=-1) #(b,t,t)
        wei=self.dropout(wei) #randomly prevent some nodes from communicating(regularisation)
        out=wei@v #(b,t,head_size)
        return out

#dropout layers are added to prevent layer overfitting(regularisation technique)
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj=nn.Linear(n_embd,n_embd) #project the back in org dimension to let them talk
        self.dropout=nn.Dropout(0.2)

    def forward(self,x):
       #find all outputs 
       head_outputs = [head(x) for head in self.heads]
       rout = torch.cat(head_outputs, dim=-1) #concat them
       rout=self.dropout(self.proj(rout)) #project back
       return rout

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout), #dropout after feed forward(andrej style)
        )

    def forward(self,x):
        return self.net(x)

#blocksof architecture
#layer norm is added to prevent the gradients from vansihing or getting two big
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa=MultiHeadAttention(n_head,head_size)
        self.ffwd=FeedForward(n_embd)
        self.ln1=nn.LayerNorm(n_embd) #layer normalisation before attention
        self.ln2=nn.LayerNorm(n_embd) #layer norm before feed forward

    def forward(self,x):
        x=x+self.sa(self.ln1(x)) #preserve the org char
        x=x+self.ffwd(self.ln2(x))
        return x


#The Biagram Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
        self.position_embedding_table=nn.Embedding(block_size,n_embd)

        n_layer=4 #how many depth of blocks want
        self.blocks=nn.Sequential(*[Block(n_embd,num_heads) for _ in range(n_layer)]) #* do unpacking as sequential wont accept a list

        self.ln_f=nn.LayerNorm(n_embd) #final layer norm
        self.lm_head=nn.Linear(n_embd,vocab_size)


    def forward(self, idx, targets=None):
        B,T=idx.shape
        tok_embd=self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))
        x=tok_embd+pos_embd
        x=self.blocks(x)
        x=self.ln_f(x) # (B,T,C)
        logits=self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #c is vocab size
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_generate):
        for _ in range(max_generate):
            idx_cond=idx[:,-block_size:] 
            logits, loss = self.forward(idx_cond)
            logits = logits[:, -1, :] # Becomes (B, C)
            prob = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(prob, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx

#Initialization & Training Loop
model = BigramLanguageModel(vocab_size)
m = model.to(device) 

# Print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print("Starting training...")
for step in range(max_iters):
    
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    xb, yb = get_batch('train')
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\n--- Training Complete! Generating Text ---\n")
# Start from the special token
start_token_id = tokenizer.token_to_id("<|endoftext|>")
context = torch.tensor([[start_token_id]], dtype=torch.long, device=device) #moving starting tensor to cuda(if avai)
generated_tokens = m.generate(context, max_generate=50)
print(decode(generated_tokens[0].tolist()))

""""Asking the model to complete"""
def ask_model(prompt, max_new_tokens=200):
    model.eval()
    start_ids = encode(prompt)
    # Convert to tensor and add a batch dimension (B=1)
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"\nPrompt: {prompt}")
    print("Response: ", end="")
    
    outcome = model.generate(x, max_generate=max_new_tokens)[0].tolist()
    
    print(decode(outcome))
    model.train()

# --- Example Usage ---
my_prompt = "Once upon a time in a land"
ask_model(my_prompt, max_new_tokens=100)

torch.save(model.state_dict(), model_path)
print(f"\nModel weights saved to {model_path}")
def load_my_model(path, vocab_size):
    # You MUST create the model architecture first
    model_to_load = BigramLanguageModel(vocab_size)
    # Then load the 'brain' into that architecture
    model_to_load.load_state_dict(torch.load(path, map_location=device))
    model_to_load.to(device)
    model_to_load.eval() # Always set to eval for inference
    return model_to_load

print("To use this model later without training, just call load_my_model().")