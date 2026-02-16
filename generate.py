import torch
import os
from model import GPT
from dataset import FineWebDataset

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'mps' if 'mps' in device else 'cpu'

# 1B-Class Modern Config
block_size = 1024
n_embd = 2048
n_head = 32
n_layer = 14
n_kv_head = 8

# Precision for inference
ptdtype = torch.bfloat16 if device_type in ['cuda', 'mps'] else torch.float32

dataset = FineWebDataset()
model = GPT(
    vocab_size=dataset.vocab_size, 
    n_embd=n_embd, 
    n_head=n_head, 
    n_layer=n_layer, 
    block_size=block_size, 
    n_kv_head=n_kv_head
).to(device)

model_path = 'model_instruct.pt' if torch.cuda.is_available() or torch.backends.mps.is_available() else 'model.pt'
# Check which model exists, prefer instruct
if not os.path.exists(model_path):
    model_path = 'model.pt'

try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Successfully loaded {model_path}")
except FileNotFoundError:
    print(f"Warning: model weights not found. Using untrained weights.")

model.eval()

# Generate from prompt
prompt = "In the future, artificial intelligence will"
chat_mode = True # Set to True to use ChatML formatting

if chat_mode:
    formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
else:
    formatted_prompt = prompt

context = torch.tensor(dataset.encode(formatted_prompt), dtype=torch.long, device=device).unsqueeze(0)

print(f"Generating (chat_mode={chat_mode})...")
with torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type != 'cpu' else torch.amp.autocast(device_type='cpu', enabled=False):
    generated = model.generate(context, max_new_tokens=150, temperature=0.8)[0].tolist()
print("-" * 30)
print(dataset.decode(generated))
print("-" * 30)
