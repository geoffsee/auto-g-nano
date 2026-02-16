import os
import json
import torch
import tiktoken
from model import GPT
from huggingface_hub import HfApi, create_repo

# Config (matching train.py)
n_embd = 2048
n_head = 32
n_layer = 14
block_size = 1024
n_kv_head = 8
dropout = 0.0
tokenizer_name = 'gpt2'

def publish_to_hf(repo_id, model_path='model.pt'):
    # 1. Get tokenizer info
    print(f"Loading tokenizer {tokenizer_name}...")
    enc = tiktoken.get_encoding(tokenizer_name)
    vocab_size = enc.n_vocab
    
    config = {
        "vocab_size": vocab_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "block_size": block_size,
        "n_kv_head": n_kv_head,
        "dropout": dropout,
        "tokenizer_name": tokenizer_name,
        "architecture": "Nemotron-style Modern Transformer",
        "weight_tying": False
    }

    # 2. Initialize and load model
    print(f"Loading 1B-class model ({n_embd} width, {n_layer} layers)...")
    model = GPT(
        vocab_size=config["vocab_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        block_size=config["block_size"],
        dropout=config["dropout"],
        n_kv_head=config["n_kv_head"]
    )
    
    # Check if model.pt exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find {model_path}. Please run train.py first.")

    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Save model and config in a format HF likes
    print("Preparing files locally in 'hf_model' folder...")
    os.makedirs("hf_model", exist_ok=True)
    
    # Save using the Mixin (handles pytorch_model.bin and config.json)
    model.save_pretrained("hf_model", config=config)
    
    # Create a better README.md (model card)
    model_card = f"""---
language: en
license: mit
datasets:
- HuggingFaceFW/fineweb-edu
tags:
- text-generation
- gpt
- nano-gpt
- pytorch
- llama-style
- rope
- gqa
- swiglu
---

# {repo_id.split('/')[-1]}

This is a modernized, 1.06B parameter decoder-only Transformer (Nemotron-style evolution) trained on the FineWeb-Edu dataset.

## Key Features
- **Nemotron-style Architecture**: Decoder-only Transformer with 4x FFN expansion and SwiGLU.
- **Modern RoPE**: Rotary positional embeddings with $\theta=500,000$ (Nemotron-4/Llama-3 standard).
- **Untied Embeddings**: Increased model capacity by decoupling input/output embeddings.
- **Grouped-Query Attention (GQA)**: Optimized for inference efficiency (3:1 query-to-KV ratio).
- **BPE Tokenization**: Uses OpenAI's `tiktoken` (GPT-2).
- **Parameters**: ~1.06B

## Model Details
- **Architecture**: Decoder-only Transformer
- **Vocab Size**: {config['vocab_size']}
- **Embedding Dimension**: {config['n_embd']}
- **Heads**: {config['n_head']}
- **KV Heads**: {config['n_kv_head']}
- **Layers**: {config['n_layer']}
- **Block Size**: {config['block_size']}

## How to Use
You can use this model directly with the `GPT` class from this repository.

```python
from model import GPT
import tiktoken
import torch

model = GPT.from_pretrained("{repo_id}")
enc = tiktoken.get_encoding("gpt2")

# Generate text
prompt = "The future of AI is"
idx = torch.tensor(enc.encode(prompt)).unsqueeze(0)
completion = model.generate(idx, max_new_tokens=50)
print(enc.decode(completion[0].tolist()))
```

## Training Data
Trained on a sample of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
"""
    with open("hf_model/README.md", "w") as f:
        f.write(model_card)
    
    # 4. Upload to Hugging Face
    print(f"Attempting to upload to https://huggingface.co/{repo_id}...")
    api = HfApi()
    
    try:
        create_repo(repo_id, exist_ok=True)
        # Upload the whole folder
        api.upload_folder(
            folder_path="hf_model",
            repo_id=repo_id,
            repo_type="model",
        )
        # Also upload model.py and dataset.py so others can load/train it
        for file in ["model.py", "dataset.py"]:
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=repo_id,
            )
        print(f"\nSuccessfully published to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"\nError uploading to HF: {e}")
        print("\nACTION REQUIRED:")
        print("1. Make sure you are logged in via `huggingface-cli login` or have HUGGING_FACE_HUB_TOKEN set.")
        print(f"2. You can also manually upload the contents of the 'hf_model' folder to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python publish.py <username>/<repo-name>")
    else:
        publish_to_hf(sys.argv[1])
