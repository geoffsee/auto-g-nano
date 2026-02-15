import torch
from model import GPT
from dataset import FineWebDataset

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Grok-style "100M-class" Config
block_size = 1024
n_embd = 768
n_head = 12
n_layer = 12
n_kv_head = 4

dataset = FineWebDataset()
model = GPT(
    vocab_size=dataset.vocab_size, 
    n_embd=n_embd, 
    n_head=n_head, 
    n_layer=n_layer, 
    block_size=block_size, 
    n_kv_head=n_kv_head
).to(device)

model_path = 'model.pt'
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded {model_path}")
except FileNotFoundError:
    print(f"Warning: {model_path} not found. Using untrained weights.")

model.eval()

# Generate from prompt
prompt = "In the future, artificial intelligence will"
context = torch.tensor(dataset.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
generated = model.generate(context, max_new_tokens=100, temperature=0.8)[0].tolist()
print("-" * 30)
print(dataset.decode(generated))
print("-" * 30)
