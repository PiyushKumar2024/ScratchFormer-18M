"""
Transformer-based Language Model architecture with multiheaded self attention,ffn, and residual connections.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from config import n_embd, num_heads, n_layer, block_size, dropout, device


# Single Attention Head
class Head(nn.Module):
    """
    One head of scaled dot-product self-attention.
    For each token, three linear projections (Key, Query, Value) are computed.
    Attention weights = softmax( Q·Kᵀ / √d_k ), then used to aggregate V.
    A causal mask prevents attending to future positions.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)  # projects input → key   space
        self.query = nn.Linear(n_embd, head_size, bias=False)  # projects input → query space
        self.value = nn.Linear(n_embd, head_size, bias=False)  # projects input → value space

        # Lower-triangular mask (True = allowed to attend). Registered as a buffer
        # so it moves to the correct device automatically but is NOT a learnable param.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # Dropout applied after softmax — randomly prevents some nodes from
        # communicating, acting as regularisation to reduce overfitting.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape                                      # (batch, time, channels)
        k = self.key(x)                                         # (B, T, head_size)
        q = self.query(x)                                       # (B, T, head_size)
        v = self.value(x)                                       # (B, T, head_size)

        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T) — raw scores
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # mask future tokens
        wei = F.softmax(wei, dim=-1)                            # (B, T, T) — normalised weights
        wei = self.dropout(wei)                                 # dropout on attention weights

        out = wei @ v                                           # (B, T, head_size) — weighted sum
        return out


# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    """
    Runs multiple attention heads in parallel, concatenates their outputs,
    then projects back to the original embedding dimension.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(n_embd, n_embd)   # linear projection back to n_embd
        self.dropout = nn.Dropout(dropout)          # dropout after projection

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]          # list of (B, T, head_size)
        concatenated = torch.cat(head_outputs, dim=-1)           # (B, T, n_embd)
        projected    = self.dropout(self.proj(concatenated))      # project + dropout
        return projected


# Position-wise Feed-Forward Network
class FeedForward(nn.Module):
    """
    Two-layer MLP applied independently to each position.
    The inner dimension is expanded to give the network more capacity for computation.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # expand
            nn.ReLU(),                         # non-linearity
            nn.Linear(4 * n_embd, n_embd),    # compress back
            nn.Dropout(dropout),               # dropout before residual connection
        )

    def forward(self, x):
        return self.net(x)


# Transformer Block
class Block(nn.Module):
    """
    One Transformer block = Communication + Computation.
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)   # self-attention
        self.ffwd = FeedForward(n_embd)                     # feed-forward
        self.ln1  = nn.LayerNorm(n_embd)                    # norm before attention
        self.ln2  = nn.LayerNorm(n_embd)                    # norm before feed-forward

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     
        x = x + self.ffwd(self.ln2(x))  
        return x


# Full Language Model
class TransformerLanguageModel(nn.Module):
    """
    Complete decoder-only Transformer language model.
    Input:   token indices  (B, T)
    Output:  logits         (B, T, vocab_size)   +   optional loss
    """

    def __init__(self, vocab_size):
        super().__init__()
        # Embedding tables
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Stack of transformer blocks (n_layer deep)
        self.blocks = nn.Sequential(
            *[Block(n_embd, num_heads) for _ in range(n_layer)]
        )

        self.ln_f    = nn.LayerNorm(n_embd)              # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)     # maps embeddings → vocab logits

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) tensor of token indices
        targets: (B, T) tensor of target token indices, or None for inference
        returns: (logits, loss)  — loss is None when targets is None
        """
        B, T = idx.shape

        tok_embd = self.token_embedding_table(idx)                            # (B, T, n_embd)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_embd + pos_embd                                                # (B, T, n_embd)

        x = self.blocks(x)         # pass through all transformer blocks
        x = self.ln_f(x)           # final layer norm                          (B, T, n_embd)
        logits = self.lm_head(x)   # project to vocab size                     (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)     # flatten for cross-entropy
            targets = targets.view(B * T)
            loss    = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_generate, temperature=1.0, top_k=None):
        """
        Autoregressively generates `max_generate` new tokens.
        idx: (B, T) tensor of starting context token indices.
        temperature: >1.0 increases randomness, <1.0 makes it more deterministic.
        top_k: if set to an integer, only sample from the top k most likely tokens.
        """
        for _ in range(max_generate):
            # Crop context to the last block_size tokens (model's max context)
            idx_cond = idx[:, -block_size:]

            logits, _ = self.forward(idx_cond)       # forward pass
            logits = logits[:, -1, :]                 # take logits of last position → (B, C)
            
            # Apply temperature scaling
            logits = logits / temperature
            
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            prob   = F.softmax(logits, dim=-1)        # convert to probabilities
            idx_next = torch.multinomial(prob, num_samples=1)  # sample next token  → (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)            # append to sequence → (B, T+1)

        return idx
