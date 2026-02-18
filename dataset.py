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
                break

    def get_batch(self, batch_size, block_size, device):
        total_required = batch_size * (block_size + 1)
        self._fill_buffer(total_required)
        
        # Select random starting indices
        start_indices = np.random.randint(0, len(self.buffer) - block_size - 1, (batch_size,))
        
        # Batching tokens more efficiently
        x_data = []
        y_data = []
        for i in start_indices:
            x_data.append(self.buffer[i : i + block_size])
            y_data.append(self.buffer[i + 1 : i + block_size + 1])
            
        # Create tensors directly from lists
        x = torch.tensor(x_data, dtype=torch.long, device=device)
        y = torch.tensor(y_data, dtype=torch.long, device=device)
        
        # Clean up buffer if it grows too large
        if len(self.buffer) > 1_000_000:
            self.buffer = self.buffer[500_000:]
            
        return x, y

    def encode(self, s):
        return self.enc.encode(s)

    def decode(self, l):
        return self.enc.decode(l)

class InstructDataset(FineWebDataset):
    """Dataset for Instruction Fine-Tuning using ChatML format and OpenHermes-2.5."""
    def __init__(self, split='train', tokenizer_name='gpt2'):
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.vocab_size = self.enc.n_vocab
        
        print(f"Loading OpenHermes-2.5 ({split})...")
        # OpenHermes-2.5 is a top-tier instruct dataset
        dataset = load_dataset("teknium/OpenHermes-2.5", split=split, streaming=True)
        
        self.dataset_iter = iter(dataset)
        self.buffer = []

    def _format_chatml(self, messages):
        """Formats a list of messages into ChatML string."""
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return formatted

    def _fill_buffer(self, n_tokens):
        while len(self.buffer) < n_tokens + 1:
            try:
                example = next(self.dataset_iter)
                # OpenHermes uses 'conversations'
                if 'conversations' in example:
                    # Convert OpenHermes format (from/value) to ChatML role/content
                    messages = []
                    for turn in example['conversations']:
                        role = turn['from']
                        if role == 'human': role = 'user'
                        if role == 'gpt': role = 'assistant'
                        messages.append({'role': role, 'content': turn['value']})
                    text = self._format_chatml(messages)
                elif 'messages' in example:
                    text = self._format_chatml(example['messages'])
                else:
                    # Fallback for completion-style examples if any
                    text = example.get('text', "")
                
                tokens = self.enc.encode_ordinary(text)
                # Append <|endoftext|>
                tokens.append(self.enc.eot_token)
                self.buffer.extend(tokens)
            except StopIteration:
                break
