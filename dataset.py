import torch

class ShakespeareDataset:
    """Minimal char-level dataset for Tiny Shakespeare."""
    def __init__(self, file_path='input.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.train_data = self.data[:int(0.9 * len(self.data))]
        self.val_data = self.data[int(0.9 * len(self.data)):]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def get_batch(self, batch_size, block_size, device, split='train'):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+1+block_size] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
