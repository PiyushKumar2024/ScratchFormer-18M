"""
train.py - Data loading, tokeniser setup, and training loop.
"""
import os
import json
import torch
from tokenizers import ByteLevelBPETokenizer

from config import (
    batch_size, block_size, max_iters, eval_interval, eval_iters,
    learning_rate, device, seed, vocab_size_limit, min_frequency,
    dataset_path, model_save_path,
)
from model import TransformerLanguageModel


#Reproducibility
torch.manual_seed(seed)


# Load Dataset
print(f"[INFO] Using device: {device}")
print(f"[INFO] Loading dataset from: {dataset_path}")

with open(dataset_path, "r", encoding="utf-8") as f:
    textData = f.read()

print(f"[INFO] Dataset length: {len(textData):,} characters")


# Tokeniser Setup (BPE)
# ByteLevelBPETokenizer from HuggingFace tokenizers library.
# It learns sub-word merges directly from the data, giving a compact
# vocabulary while still being able to represent any byte sequence.
SPECIAL_TOKEN = "<|endoftext|>"

tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(
    [textData],                       # training data
    vocab_size=vocab_size_limit,      # maximum vocabulary size
    min_frequency=min_frequency,      # minimum token frequency to keep
    special_tokens=[SPECIAL_TOKEN],   # end-of-text marker
)

vocab_size = tokenizer.get_vocab_size()
print(f"[INFO] Vocabulary size: {vocab_size}")


# Helper functions for encoding / decoding text
def encode(text):
    """Convert a string to a list of token IDs."""
    return tokenizer.encode(text).ids

def decode(token_ids):
    """Convert a list of token IDs back to a string."""
    return tokenizer.decode(token_ids)


# Data Splitting and Batching
# Encode the entire dataset into a single long tensor of token IDs
data = torch.tensor(encode(textData), dtype=torch.long)

# 85 / 15 split for training / validation
n = int(len(data) * 0.85)
train_data = data[:n]
val_data   = data[n:]

print(f"[INFO] Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")


def get_batch(split):
    """
    Sample a random batch of (input, target) pairs.
    Each sample is a chunk of block_size consecutive tokens.
    The target is the same chunk shifted right by one position.
    """
    data_source = train_data if split == "train" else val_data

    # Random starting indices for each sample in the batch
    rand_starts = torch.randint(0, len(data_source) - block_size, (batch_size,))

    x = torch.stack([data_source[i   : i + block_size]     for i in rand_starts])
    y = torch.stack([data_source[i+1 : i + block_size + 1] for i in rand_starts])

    # Move tensors to GPU if available
    x, y = x.to(device), y.to(device)
    return x, y


# Loss Estimation
@torch.no_grad()  # disable gradient tracking to save memory
def estimate_loss():
    """
    Estimate train and val loss by averaging over eval_iters batches.
    Using multiple batches gives a smoother, more reliable loss estimate
    than looking at a single batch.
    """
    out = {}
    model.eval()  # switch to eval mode (disables dropout)
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()  # switch back to training mode (re-enables dropout)
    return out


# Model Init
model = TransformerLanguageModel(vocab_size)
model = model.to(device)

# Print total parameter count
param_count = sum(p.numel() for p in model.parameters()) / 1e6
print(f"[INFO] Model parameters: {param_count:.3f}M")


#  Training Loop
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"\n{'='*60}")
print(f"  Starting training for {max_iters:,} steps")
print(f"{'='*60}\n")

best_val_loss = float('inf')
checkpoint_dir = os.path.dirname(model_save_path) or "."

for step in range(max_iters):
    # Periodically evaluate, print losses, and save checkpoint
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print(f"  step {step:>5d}  |  train loss: {losses['train']:.4f}  |  val loss: {losses['val']:.4f}")

        # Save checkpoint after every eval
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_step{step:05d}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"           -> checkpoint saved: {ckpt_path}")

        # Also save as "best" if this is the lowest val loss so far
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"           -> new best model! (val loss: {best_val_loss:.4f})")

    # Sample a batch and compute loss
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    # Backprop and update weights
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"\n{'='*60}")
print("  Training complete!")
print(f"{'='*60}\n")


# Quick Generation Test
start_token_id = tokenizer.token_to_id(SPECIAL_TOKEN)
context = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
generated_tokens = model.generate(context, max_generate=50)
print("[Sample output]:")
print(decode(generated_tokens[0].tolist()))


# Save Model and Tokeniser
torch.save(model.state_dict(), model_save_path)
print(f"\n[INFO] Model weights saved to: {model_save_path}")

# Save tokeniser files (vocab.json + merges.txt) so run.py can reload it
tokenizer_dir = "tokenizer"
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer.save_model(tokenizer_dir)
print(f"[INFO] Tokeniser saved to: {tokenizer_dir}/")

# Also save vocab_size so run.py knows the exact number
meta = {"vocab_size": vocab_size}
with open(os.path.join(tokenizer_dir, "meta.json"), "w") as f:
    json.dump(meta, f)

print("\n[DONE] Run  python run.py  to generate text from the trained model.")
