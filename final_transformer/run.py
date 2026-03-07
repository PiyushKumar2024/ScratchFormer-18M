"""
run.py - Load a trained model and generate text interactively.

Usage:
    python run.py                        # generates from the special start token
    python run.py --prompt "Once upon"   # completes from the given prompt
    python run.py --tokens 200           # controls how many tokens to generate

Prerequisites:
    - A trained model saved by train.py  (transformer_model.pth)
    - Tokeniser files saved by train.py  (tokenizer/ folder)
"""
import os
import json
import argparse
import torch
from tokenizers import ByteLevelBPETokenizer

from config import block_size, device, model_save_path
from model import TransformerLanguageModel


# Argument Parsing from command line
parser = argparse.ArgumentParser(description="Generate text from a trained Transformer.")
parser.add_argument("--prompt",      type=str, default=None,  help="Starting text to continue from.")
parser.add_argument("--tokens",      type=int, default=200,   help="Number of tokens to generate.")
parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for generation (lower = less random).")
parser.add_argument("--top_k",       type=int, default=40,    help="Top-K sampling (only pick from K most likely).")
parser.add_argument("--model",       type=str, default=model_save_path, help="Path to model weights.")
args = parser.parse_args()


# ========================== Load Tokeniser ===============================
tokenizer_dir = "tokenizer"
SPECIAL_TOKEN = "<|endoftext|>"

# Check that tokeniser files exist
vocab_path  = os.path.join(tokenizer_dir, "vocab.json")
merges_path = os.path.join(tokenizer_dir, "merges.txt")
meta_path   = os.path.join(tokenizer_dir, "meta.json")

if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
    print("[ERROR] Tokeniser files not found. Run train.py first.")
    exit(1)

tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)

# Load vocab size from metadata
with open(meta_path, "r") as f:
    meta = json.load(f)
vocab_size = meta["vocab_size"]

print(f"[INFO] Tokeniser loaded  |  vocab size: {vocab_size}")


def encode(text):
    """Convert a string to a list of token IDs."""
    return tokenizer.encode(text).ids

def decode(token_ids):
    """Convert a list of token IDs back to a string."""
    return tokenizer.decode(token_ids)


# Load Model 
print(f"[INFO] Loading model from: {args.model}")

model = TransformerLanguageModel(vocab_size)
model.load_state_dict(torch.load(args.model, map_location=device))
model.to(device)
model.eval()  # set to evaluation mode (disables dropout)

param_count = sum(p.numel() for p in model.parameters()) / 1e6
print(f"[INFO] Model loaded  |  {param_count:.3f}M parameters  |  device: {device}")


# Generate Text 
def generate_text(prompt=None, max_new_tokens=200):
    """
    Generate text either from a prompt or from the special start token.
    """
    if prompt is not None:
        # Encode the user's prompt as the starting context
        start_ids = encode(prompt)
        context = torch.tensor([start_ids], dtype=torch.long, device=device)
        print(f"\nPrompt: {prompt}")
    else:
        # Start from the special end-of-text token
        start_token_id = tokenizer.token_to_id(SPECIAL_TOKEN)
        context = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
        print("\n[Generating from start token]")

    print("-" * 60)

    with torch.no_grad():
        generated = model.generate(
            context,
            max_generate=max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )

    output_text = decode(generated[0].tolist())
    print(output_text)
    print("-" * 60)

    return output_text


# Main 
if __name__ == "__main__":
    generate_text(prompt=args.prompt, max_new_tokens=args.tokens)

    # Interactive mode: keep asking for prompts
    print("\n[Interactive Mode] Type a prompt and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input(">>> ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if user_input == "":
                generate_text(prompt=None, max_new_tokens=args.tokens)
            else:
                generate_text(prompt=user_input, max_new_tokens=args.tokens)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
