import torch
from model import GPT
from dataset import ShakespeareDataset

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
dataset = ShakespeareDataset('../input.txt')
model = GPT(dataset.vocab_size).to(device)  # uses defaults
model.load_state_dict(torch.load('../model.pt'))
model.eval()

# Generate from prompt
prompt = "\n"  # or "ROMEO:"
context = torch.tensor(dataset.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
generated = model.generate(context, max_new_tokens=500, temperature=0.8)[0].tolist()
print(dataset.decode(generated))
