import torch
import math
import time
from model import GPT
from dataset import InstructDataset

# --- 1B-Class Fine-Tuning Config ---
batch_size = 1         
gradient_accumulation_steps = 128 
block_size = 1024     
n_embd = 2048
n_head = 32
n_layer = 14
n_kv_head = 8         
dropout = 0.0
learning_rate = 5e-5  # Lower LR for fine-tuning
weight_decay = 0.1
max_iters = 5000      # SFT is typically much shorter than pre-training
eval_interval = 250
eval_iters = 10
warmup_iters = 100
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'mps' if 'mps' in device else 'cpu'

# Precision
if device_type == 'cuda':
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
elif device_type == 'mps':
    dtype = 'bfloat16'
else:
    dtype = 'float32'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type != 'cpu' else torch.amp.autocast(device_type='cpu', enabled=False)

# Setup Dataset
train_dataset = InstructDataset(split='train')
val_dataset = InstructDataset(split='train') # Using small slice of train for val simplicity

model = GPT(
    vocab_size=train_dataset.vocab_size, 
    n_embd=n_embd, 
    n_head=n_head, 
    n_layer=n_layer, 
    block_size=block_size, 
    dropout=dropout,
    n_kv_head=n_kv_head
).to(device)

# Load pre-trained weights if available
model_path = 'model.pt'
try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded pre-trained model from {model_path} for fine-tuning.")
except FileNotFoundError:
    print(f"Warning: {model_path} not found. Starting fine-tuning from scratch (not recommended).")

# Optimizer
param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return learning_rate * 0.1
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate * (0.1 + 0.9 * coeff)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {'train': 0.0, 'val': 0.0}
    for split, ds in [('train', train_dataset), ('val', val_dataset)]:
        for _ in range(eval_iters):
            X, Y = ds.get_batch(batch_size, block_size, device)
            with ctx:
                _, loss = model(X, Y)
            losses[split] += loss.item()
        losses[split] /= eval_iters
    model.train()
    return losses

print(f"Starting Instruction Fine-Tuning on {device} ({dtype})...")
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))

t0 = time.time()
for iter in range(max_iters):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter:5d} | lr {lr:.2e} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(gradient_accumulation_steps):
        xb, yb = train_dataset.get_batch(batch_size, block_size, device)
        with ctx:
            logits, loss = model(xb, yb)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
    
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    if iter % 10 == 0 and iter > 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        tokens_processed = batch_size * block_size * gradient_accumulation_steps * 10
        tokens_per_sec = tokens_processed / dt
        print(f"iter {iter:5d} | loss {loss.item()*gradient_accumulation_steps:.4f} | {tokens_per_sec:.0f} tok/s")

torch.save(model.state_dict(), 'model_instruct.pt')
print("Fine-tuning complete! Model saved as model_instruct.pt")
