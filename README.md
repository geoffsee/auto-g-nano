# auto-g-nano

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/models?search=auto-g-nano)

A complete, minimal, and fully functional decoder-only Language Model (nanoGPT-style) built from absolute scratch in pure PyTorch. No high-level abstractions, no pre-built modules—just the essentials of the Transformer architecture.

---

## 📖 Table of Contents
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Inference](#-inference)
- [Hugging Face Integration](#-hugging-face-integration)
- [Results](#-results)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## ✨ Features
- **Pure PyTorch**: Implemented using basic `nn.Module` and `functional` operations.
- **NanoGPT Style**: Focused on simplicity and readability, ideal for learning.
- **Character-level Tokenization**: Simple and effective for small datasets like Tiny Shakespeare.
- **Hugging Face Hub Support**: Built-in scripts to push and pull models from the Hub.
- **Modern Tooling**: Managed with `uv` for fast, reproducible environments.

## 🏗 Architecture
The model is a standard decoder-only Transformer with:
- **Token & Positional Embeddings**
- **Causal Self-Attention**: Multi-head attention with masking to prevent looking into the future.
- **Feed-Forward Networks**: Simple 2-layer MLP with ReLU activation.
- **Layer Normalization**: Applied before each sub-layer (Pre-Norm).
- **Residual Connections**: Crucial for deep network stability.

| Hyperparameter | Value |
| :--- | :--- |
| Parameters | ~10.8M |
| Context Length | 256 |
| Embedding Dim | 384 |
| Heads | 6 |
| Layers | 6 |
| Dropout | 0.2 |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended)

### Installation
```bash
git clone https://github.com/your-username/auto-g-nano.git
cd auto-g-nano
uv sync
```

### Download Dataset
Fetch the Tiny Shakespeare dataset:
```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

---

## 🏋️ Training
Start training the model on the character-level dataset:
```bash
uv run python train.py
```
The script will automatically use `cuda`, `mps`, or `cpu` based on availability. The trained weights are saved to `model.pt`.

---

## 🔮 Inference
Generate text using the trained model:
```bash
uv run python generate.py
```
Sample output should resemble Shakespearean dialogue after sufficient training.

---

## 🤗 Hugging Face Integration

### Publishing
Upload your model and code to the Hugging Face Hub:
```bash
uv run python publish.py <your-username>/<your-model-name>
```

### Loading
Load a published model directly in your code:
```python
from model import GPT
model = GPT.from_pretrained("<your-username>/<your-model-name>")
```

---

## 📊 Results

### Training Curve
Expect the loss to start around ~4.3 and drop to ~1.5 after 5000 iterations.
```console
step    0 | train loss 4.2932 | val loss 4.2972
step 1000 | train loss 1.3553 | val loss 1.5904
step 2500 | train loss 1.0986 | val loss 1.4848
step 5000 | train loss 0.8442 | val loss 1.5871
```

### Sample Output
> **LADY GREY:**
> And their I know before I descend.
> 
> **KING EDWARD IV:**
> But what prophecy they love to do thee for thee?
> 
> **GLOUCESTER:**
> The bridge makes her majesty to be at once.

---

## 🙏 Acknowledgments
- Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).
- Dataset provided by the Tiny Shakespeare corpus.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.