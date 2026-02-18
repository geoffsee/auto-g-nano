# auto-g-nano 🚀

`auto-g-nano` is a high-performance, modern implementation of a decoder-only Transformer (GPT) designed for efficiency and speed. While it draws inspiration from projects like nanoGPT, it incorporates several state-of-the-art architectural improvements and is specifically optimized for **Apple Silicon (M-series)** hardware.

The default configuration is tuned for a **~1.06 billion parameter** model, but it can be easily scaled up or down.

## ✨ Key Features

- **Modern Architecture**:
  - **Grouped-Query Attention (GQA)**: Reduces memory bandwidth overhead during inference (3:1 query-to-KV ratio).
  - **Rotary Positional Embeddings (RoPE)**: Nemotron-4/Llama-3 standard with $\theta=500,000$.
  - **RMSNorm**: Faster and more stable layer normalization.
  - **SwiGLU FFN**: Nemotron-style feed-forward network with 4x expansion.
  - **Untied Embeddings**: Decoupled input/output embeddings for increased capacity.
- **Apple Silicon Optimized**:
  - **PyTorch MPS Support**: Native acceleration on Mac GPUs with optimized batching for unified memory.
  - **Native MLX Implementation**: Built using Apple's [MLX](https://github.com/ml-explore/mlx) framework for maximum throughput on M-series chips.
  - **CoreML Export**: Convert models for high-efficiency inference on the Apple Neural Engine (ANE).
- **Comprehensive Pipeline**:
  - **Pre-training**: Stream datasets directly from FineWeb-Edu using `tiktoken` (BPE).
  - **Instruction Fine-Tuning**: Fine-tune on OpenHermes-2.5 using the ChatML format.
  - **Evaluation**: Built-in HellaSwag benchmark script.
  - **Publishing**: Easy one-script upload to the Hugging Face Hub.

## 🛠️ Installation

This project uses `uv` for lightning-fast dependency management.

```bash
# Clone the repository
git clone https://github.com/geoffsee/auto-g-nano.git
cd auto-g-nano

# Install dependencies
uv sync
```

## 🚀 Usage

### 1. Pre-training (PyTorch)
Train the model on the FineWeb-Edu dataset using PyTorch. Supports MPS (Mac), CUDA (NVIDIA), and CPU.
```bash
uv run train.py
```
*Note: Configured for 1.06B parameters by default. Adjust `n_embd`, `n_head`, and `n_layer` in `train.py` to change model size.*

### 2. Native Apple Silicon Training (MLX)
For the best performance on M-series chips, use the MLX training script. It utilizes unified memory and lazy evaluation for significant speedups over standard MPS.
```bash
uv run mlx_train.py
```

### 3. Instruction Fine-Tuning
Fine-tune a pre-trained model (e.g., `model.pt`) on the OpenHermes-2.5 dataset using the ChatML format.
```bash
uv run finetune.py
```

### 4. Generation
Generate text using your trained model. Supports both completion and ChatML modes.
```bash
uv run generate.py
```

### 5. Evaluation
Test your model's zero-shot performance on the HellaSwag benchmark.
```bash
uv run eval_hellaswag.py 500  # Run on 500 samples
```

### 6. Publishing to Hugging Face
Upload your model weights, config, and README to the Hugging Face Hub.
```bash
uv run publish.py <username>/<repo-name>
```

## 🍏 Apple Neural Engine (ANE) & CoreML

Export your model to CoreML to leverage the Apple Neural Engine for low-power, high-speed inference on Mac, iPhone, and iPad.

```bash
uv run coreml_export.py
```
This generates a `.mlpackage` that can be integrated into macOS/iOS apps or used for local inference via `coremltools`.

## 📂 Project Structure

- `model.py`: PyTorch implementation of the modern GPT architecture.
- `mlx_model.py`: MLX implementation for Apple Silicon.
- `train.py` / `mlx_train.py`: Training scripts for PyTorch and MLX.
- `finetune.py`: Instruction fine-tuning script.
- `dataset.py`: Data loading and tokenization (FineWeb-Edu & OpenHermes-2.5).
- `generate.py`: Text generation script.
- `coreml_export.py`: CoreML conversion utility.
- `eval_hellaswag.py`: HellaSwag evaluation suite.
- `publish.py`: Hugging Face Hub publishing utility.

## 📜 License
MIT
