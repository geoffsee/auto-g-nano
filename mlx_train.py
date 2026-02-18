import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import math
import numpy as np
from mlx_model import GPT
from dataset import FineWebDataset

# --- 1B-Class Modern Config (MLX Optimized) ---
batch_size = 16          # MLX handles larger batches well on Unified Memory
gradient_accumulation_steps = 8 # Total effective batch = 128
block_size = 1024
n_embd = 2048
n_head = 32
n_layer = 14
n_kv_head = 8
dropout = 0.0
learning_rate = 3e-4
weight_decay = 0.1
max_iters = 50000
eval_interval = 500
eval_iters = 10
warmup_iters = 1000

# Setup Dataset
train_dataset = FineWebDataset(split='train')
val_dataset = FineWebDataset(split='train')

# Initialize Model
model = GPT(
    vocab_size=train_dataset.vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    n_kv_head=n_kv_head,
    dropout=dropout,
    block_size=block_size
)
def count_parameters(model):
    def _count_recursive(obj):
        if isinstance(obj, mx.array):
            return obj.size
        elif isinstance(obj, dict):
            return sum(_count_recursive(v) for v in obj.values())
        elif isinstance(obj, list):
            return sum(_count_recursive(v) for v in obj)
        return 0
    return _count_recursive(model.parameters())

print(f"MLX Model created with {count_parameters(model)/1e6:.1f}M params")

# Optimizer
optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return learning_rate * 0.1
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate * (0.1 + 0.9 * coeff)

def loss_fn(x, y):
    logits, loss = model(x, y)
    return loss

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

@mx.compile
def train_step(x, y):
    loss, grads = loss_and_grad_fn(x, y)
    return loss, grads

def get_batch_mlx(ds, batch_size, block_size):
    # Reuse the PyTorch-based dataset but convert to MLX
    x_pt, y_pt = ds.get_batch(batch_size, block_size, device='cpu')
    return mx.array(x_pt.numpy()), mx.array(y_pt.numpy())

# Training Loop
print("Starting MLX training on Apple Silicon SOC...")
t0 = time.time()

for iter in range(max_iters):
    # Update LR
    lr = get_lr(iter)
    optimizer.learning_rate = lr

    # Eval
    if iter % eval_interval == 0:
        model.eval()
        val_loss = 0
        for _ in range(eval_iters):
            x, y = get_batch_mlx(val_dataset, batch_size, block_size)
            val_loss += loss_fn(x, y).item()
        val_loss /= eval_iters
        print(f"step {iter:5d} | lr {lr:.2e} | val loss {val_loss:.4f}")
        model.train()

    # Training step with grad accumulation
    accumulated_loss = 0
    
    for _ in range(gradient_accumulation_steps):
        x, y = get_batch_mlx(train_dataset, batch_size, block_size)
        loss, grads = train_step(x, y)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)
        accumulated_loss += loss.item()

    if iter % 10 == 0 and iter > 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        tokens_processed = batch_size * block_size * gradient_accumulation_steps * 10
        tokens_per_sec = tokens_processed / dt
        print(f"iter {iter:5d} | loss {accumulated_loss/gradient_accumulation_steps:.4f} | {tokens_per_sec:.0f} tok/s")

# Save model
mx.savez("model_mlx.npz", **dict(model.parameters()))
print("Training complete! MLX model saved.")
