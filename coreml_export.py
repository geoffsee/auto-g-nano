import torch
import coremltools as ct
from model import GPT

def export_to_coreml(model_path='model.pt'):
    # Model config
    n_embd = 2048
    n_head = 32
    n_layer = 14
    n_kv_head = 8
    block_size = 1024
    vocab_size = 50257 # Tiktoken GPT-2 vocab

    print("Loading model for CoreML export...")
    model = GPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        n_kv_head=n_kv_head,
        block_size=block_size
    )
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except FileNotFoundError:
        print(f"Warning: {model_path} not found. Exporting empty model.")
    
    model.eval()

    # Create dummy input for tracing
    # Note: CoreML prefers fixed shapes for ANE optimization
    dummy_input = torch.randint(0, vocab_size, (1, block_size))
    
    print("Tracing model with TorchScript...")
    traced_model = torch.jit.trace(model, (dummy_input,))

    print("Converting to CoreML (Optimized for Apple Silicon / Neural Engine)...")
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_ids", shape=dummy_input.shape, dtype=int)],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.ALL, # Allows ANE utilization
    )

    coreml_model.save("auto-g-nano.mlpackage")
    print("Export complete! CoreML model saved as auto-g-nano.mlpackage")
    print("This model is now ready for hardware-accelerated inference on the Apple Neural Engine.")

if __name__ == "__main__":
    export_to_coreml()
