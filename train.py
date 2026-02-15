import torch
from model import GPT
from dataset import ShakespeareDataset

# Config
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 250
eval_iters = 200
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# Setup
dataset = ShakespeareDataset('input.txt')
model = GPT(dataset.vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {'train': 0.0, 'val': 0.0}
    for split in ['train', 'val']:
        for _ in range(eval_iters):
            X, Y = dataset.get_batch(batch_size, block_size, device, split)
            _, loss = model(X, Y)
            losses[split] += loss.item()
        losses[split] /= eval_iters
    model.train()
    return losses

# Train
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    xb, yb = dataset.get_batch(batch_size, block_size, device, 'train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'model.pt')
print("Training complete! Model saved.")
