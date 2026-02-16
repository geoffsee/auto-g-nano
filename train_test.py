import torch
import math
import time
from model import GPT
from dataset import FineWebDataset

# --- Grok-style "100M-class" Config ---
batch_size = 12       # Small micro-batch for local machines
gradient_accumulation_steps = 8 # Effective batch size = 96
block_size = 1024     # Modern context length
n_embd = 768
n_head = 12
n_layer = 12
n_kv_head = 4         # GQA: 3 query heads per KV head
dropout = 0.0         # Modern LLMs often use 0 dropout if data is sufficient
learning_rate = 6e-4
weight_decay = 0.1
max_iters = 20     # Scale as needed
eval_interval = 10
eval_iters = 50
warmup_iters = 200    # Cosine warmup
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for autocast
# Use bfloat16 if available (A100+, H100), otherwise float16 (V100, T4, consumer GPUs)
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type != 'cpu' else torch.amp.autocast(device_type='cpu', enabled=False)

# Setup
train_dataset = FineWebDataset(split='train')
val_dataset = FineWebDataset(split='train') 

model = GPT(
    vocab_size=train_dataset.vocab_size, 
    n_embd=n_embd, 
    n_head=n_head, 
    n_layer=n_layer, 
    block_size=block_size, 
    dropout=dropout,
    n_kv_head=n_kv_head
).to(device)

# Weight decay logic: only for 2D parameters (weights), not 1D (biases, norms)
param_dict = {pn: p for pn, p in model.named_parameters()}
param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))

# Cosine learning rate scheduler with warmup
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

# Training Loop
print(f"Starting Grok-style training on {device} ({dtype})...")
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

t0 = time.time()
for iter in range(max_iters):
    # Update LR
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Eval
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter:5d} | lr {lr:.2e} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    # Forward/Backward
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(gradient_accumulation_steps):
        xb, yb = train_dataset.get_batch(batch_size, block_size, device)
        with ctx:
            logits, loss = model(xb, yb)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
    
    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Step
    scaler.step(optimizer)
    scaler.update()

    # Log performance
    if iter % 10 == 0 and iter > 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        tokens_per_sec = (batch_size * block_size * gradient_accumulation_steps * 10) / dt
        print(f"iter {iter:5d} | loss {loss.item()*gradient_accumulation_steps:.4f} | {tokens_per_sec:.0f} tok/s")

torch.save(model.state_dict(), 'model.pt')
print("Training complete! Model saved.")
