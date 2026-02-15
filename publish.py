import os
import json
import torch
from model import GPT
from dataset import ShakespeareDataset
from huggingface_hub import HfApi, create_repo

# Config (matching train.py)
n_embd = 384
n_head = 6
n_layer = 6
block_size = 256
dropout = 0.2

def publish_to_hf(repo_id, model_path='model.pt', dataset_path='input.txt'):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Could not find {dataset_path}. The dataset is needed to reconstruct the vocabulary for publishing.")
    
    print("Loading dataset for vocab...")
    dataset = ShakespeareDataset(dataset_path)
    config = {
        "vocab_size": dataset.vocab_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "block_size": block_size,
        "dropout": dropout,
        "stoi": dataset.stoi,
        "itos": dataset.itos
    }

    # 2. Initialize and load model
    print("Loading model...")
    model = GPT(
        vocab_size=config["vocab_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        block_size=config["block_size"],
        dropout=config["dropout"]
    )
    
    # Check if model.pt exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find {model_path}. Please run train.py first.")

    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)

    # 3. Save model and config in a format HF likes
    print("Preparing files for upload...")
    os.makedirs("hf_model", exist_ok=True)
    
    # Save using the Mixin
    model.save_pretrained("hf_model", config=config)
    
    # Create a better README.md (model card)
    model_card = f"""---
language: en
license: mit
datasets:
- tinyshakespeare
tags:
- text-generation
- gpt
- nano-gpt
- pytorch
---

# {repo_id.split('/')[-1]}

This is a minimal, decoder-only Transformer (nanoGPT-style) trained from scratch on the Tiny Shakespeare dataset.

## Model Details
- **Architecture**: Decoder-only Transformer
- **Parameters**: ~10.8M
- **Vocabulary Size**: {config['vocab_size']}
- **Embedding Dimension**: {config['n_embd']}
- **Heads**: {config['n_head']}
- **Layers**: {config['n_layer']}
- **Block Size**: {config['block_size']}

## How to Use
You can use this model directly with the `GPT` class from this repository.

```python
from model import GPT

model = GPT.from_pretrained("{repo_id}")
# Generate text
# context = torch.zeros((1, 1), dtype=torch.long)
# print(model.generate(context, max_new_tokens=100))
```

## Training Data
Trained on the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset.
"""
    with open("hf_model/README.md", "w") as f:
        f.write(model_card)
    
    # Also save the vocab explicitly just in case
    with open("hf_model/config.json", "w") as f:
        json.dump(config, f, indent=4)

    # 4. Upload to Hugging Face
    print(f"Uploading to {repo_id}...")
    api = HfApi()
    
    try:
        create_repo(repo_id, exist_ok=True)
        api.upload_folder(
            folder_path="hf_model",
            repo_id=repo_id,
            repo_type="model",
        )
        # Also upload model.py so others can load it
        api.upload_file(
            path_or_fileobj="model.py",
            path_in_repo="model.py",
            repo_id=repo_id,
        )
        print(f"Successfully published to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Error uploading to HF: {e}")
        print("Make sure you are logged in via `huggingface-cli login` or have HUGGING_FACE_HUB_TOKEN set.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python publish.py <username>/<repo-name>")
    else:
        publish_to_hf(sys.argv[1])
