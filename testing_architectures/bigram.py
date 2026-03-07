import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer

# Hyperparameters
batch_size = 32
block_size = 32
max_iters = 20000
eval_interval = 1000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

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

#The Bigram Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Table creation
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensors of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_generate):
        for _ in range(max_generate):
            # Pass the current sequence to get predictions
            logits, loss = self.forward(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :] # Becomes (B, C)
            # Apply softmax to get probabilities
            prob = F.softmax(logits, dim=-1) 
            # Sample from the distribution
            idx_next = torch.multinomial(prob, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx

#Initialization & Training Loop
model = BigramLanguageModel(vocab_size)
m = model.to(device) # Move model to GPU if available

# Print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# AdamW Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting training...")
for step in range(max_iters):
    
    # Every once in a while, evaluate the loss on train and val sets
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    # Sample a batch of data
    xb, yb = get_batch('train')
    
    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#Final Generation
print("\n--- Training Complete! Generating Text ---\n")
# Start from the special token
start_token_id = tokenizer.token_to_id("<|endoftext|>")
context = torch.tensor([[start_token_id]], dtype=torch.long, device=device) #moving starting tensor to cuda(if avai)

# Generate 500 new tokens
generated_tokens = m.generate(context, max_generate=50)
print(decode(generated_tokens[0].tolist()))