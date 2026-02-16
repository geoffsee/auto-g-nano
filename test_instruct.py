from dataset import InstructDataset
import torch

def test_instruct_dataset():
    ds = InstructDataset(split='train')
    print("Loading batch...")
    x, y = ds.get_batch(batch_size=2, block_size=128, device='cpu')
    print(f"X shape: {x.shape}")
    print(f"Y shape: {y.shape}")
    
    # Decode first example
    decoded = ds.decode(x[0].tolist())
    print("-" * 20)
    print("Decoded sample (start):")
    print(decoded[:500])
    print("-" * 20)
    
    # Check for ChatML tokens
    if "<|im_start|>" in decoded:
        print("SUCCESS: ChatML tokens found in dataset output.")
    else:
        print("WARNING: ChatML tokens NOT found in dataset output. Check _format_chatml.")

if __name__ == "__main__":
    test_instruct_dataset()
