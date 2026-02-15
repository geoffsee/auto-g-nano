import torch
import tiktoken
from datasets import load_dataset
import numpy as np

class FineWebDataset:
    """Dataset class that uses tiktoken for BPE and streams FineWeb-Edu."""
    def __init__(self, split='train', tokenizer_name='gpt2'):
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.vocab_size = self.enc.n_vocab
        
        # We use a small subset of FineWeb-Edu for the 'setup'
        # In a real training run, you'd use a much larger slice or stream it
        print(f"Loading FineWeb-Edu ({split})...")
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split=split, streaming=True)
        
        # Buffer some data for efficiency
        self.dataset_iter = iter(dataset)
        self.buffer = []
        
    def _fill_buffer(self, n_tokens):
        while len(self.buffer) < n_tokens + 1:
            try:
                example = next(self.dataset_iter)
                tokens = self.enc.encode_ordinary(example['text'])
                # Append <|endoftext|> (approximate)
                tokens.append(self.enc.eot_token)
                self.buffer.extend(tokens)
            except StopIteration:
                # Loop dataset if we run out (for simplicity in this script)
                # In real scenarios, you'd handle multiple epochs properly
                break

    def get_batch(self, batch_size, block_size, device):
        # Ensure we have enough tokens for the batch
        total_required = batch_size * (block_size + 1)
        self._fill_buffer(total_required)
        
        # Grab a random starting point in the buffer
        # (Very simplified batching for demonstration)
        start_indices = torch.randint(0, len(self.buffer) - block_size - 1, (batch_size,))
        
        x_list = []
        y_list = []
        for i in start_indices:
            chunk = self.buffer[i:i+block_size+1]
            x_list.append(torch.tensor(chunk[:-1], dtype=torch.long))
            y_list.append(torch.tensor(chunk[1:], dtype=torch.long))
            
        x = torch.stack(x_list).to(device)
        y = torch.stack(y_list).to(device)
        
        # Cleanup buffer slightly to prevent it growing infinitely
        if len(self.buffer) > 1_000_000:
            self.buffer = self.buffer[500_000:]
            
        return x, y

    def encode(self, s):
        return self.enc.encode(s)

    def decode(self, l):
        return self.enc.decode(l)
