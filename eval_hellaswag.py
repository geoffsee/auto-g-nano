import torch
import torch.nn.functional as F
from model import GPT
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def evaluate_hellaswag(model_path, num_samples=None):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Config (must match training)
    n_embd = 768
    n_head = 12
    n_layer = 12
    block_size = 1024
    n_kv_head = 4
    tokenizer_name = 'gpt2'

    enc = tiktoken.get_encoding(tokenizer_name)
    vocab_size = enc.n_vocab

    print("Loading model...")
    model = GPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        n_kv_head=n_kv_head,
        dropout=0.0
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print("Loading HellaSwag dataset...")
    dataset = load_dataset("hellaswag", split="validation", trust_remote_code=True)
    
    if num_samples is not None:
        # Use a subset for faster evaluation
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    correct = 0
    total = 0

    with torch.no_grad():
        for example in tqdm(dataset):
            context = example['ctx']
            label = int(example['label'])
            endings = example['endings']
            
            context_tokens = enc.encode(context)
            
            # We want to find which ending has the highest probability given the context
            log_probs = []
            
            for ending in endings:
                # Add a space before ending
                full_text = context + " " + ending
                full_tokens = enc.encode(full_text)
                ctx_tokens = enc.encode(context)
                
                # The number of tokens we want to predict is everything after the context
                # However, BPE can change the last token of context when merged with ending
                # So we use the length difference
                num_ending_tokens = len(full_tokens) - len(ctx_tokens)
                
                if len(full_tokens) > block_size:
                    full_tokens = full_tokens[-block_size:]
                    # adjust num_ending_tokens if it was cut off
                    # (unlikely for HellaSwag but good for safety)
                
                x = torch.tensor(full_tokens[:-1]).unsqueeze(0).to(device)
                y = torch.tensor(full_tokens[1:]).unsqueeze(0).to(device)
                
                logits, _ = model(x)
                
                # We want the log_probs of the tokens in y that correspond to the ending
                relevant_logits = logits[0, -num_ending_tokens:, :]
                relevant_targets = y[0, -num_ending_tokens:]
                
                log_p = F.log_softmax(relevant_logits, dim=-1)
                target_log_probs = log_p.gather(1, relevant_targets.unsqueeze(1)).squeeze(1)
                
                log_probs.append(target_log_probs.sum().item())
                
            prediction = torch.tensor(log_probs).argmax().item()
            if total < 5:
                print(f"\nContext: {context}")
                print(f"Endings: {endings}")
                print(f"Prediction: {prediction}, Label: {label}")
                print(f"Log probs: {log_probs}")
            
            if prediction == label:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\nHellaSwag Accuracy: {accuracy:.4%}")
    print(f"Correct: {correct}/{total}")
    return accuracy

if __name__ == "__main__":
    import sys
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    evaluate_hellaswag("model.pt", num_samples=samples)
