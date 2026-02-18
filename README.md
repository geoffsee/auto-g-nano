# auto-g-nano

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/models?search=auto-g-nano)

A complete, minimal, and fully functional decoder-only Language Model (nanoGPT-style) built from absolute scratch in pure PyTorch. No high-level abstractions, no pre-built modules—just the essentials of the Transformer architecture.

---

## 📖 Table of Contents
- [About](#about)
- [Models](#-models)
- [Architecture](#-architecture)
- [Getting Started](#-getting-started)
- [License](#-license)

---

## About

This repository contains multiple trained decoder-only Transformer models of varying sizes, all trained on the Tiny Shakespeare dataset using character-level tokenization. Each model is implemented in pure PyTorch from scratch with no high-level abstractions—just the fundamentals of the Transformer architecture.

For implementation details, training scripts, and inference code, check out the branch corresponding to your desired model size.

---

## 🤖 Models

All models are available on the Hugging Face Hub and in git branches:

| Model | Parameters | Context | Embedding | Heads | Layers | Branch | HF Hub |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **auto-g-nano** | 10.8M | 256 | 384 | 6 | 6 | [`auto-g-nano-10m`](https://github.com/williamseemueller/auto-g-nano/tree/auto-g-nano-10m) | [🤗 Hub](https://huggingface.co/models?search=auto-g-nano) |
| **auto-g-nano-153M** | 153M | 256 | 768 | 12 | 12 | [`auto-g-nano-153M`](https://github.com/williamseemueller/auto-g-nano/tree/auto-g-nano-153M) | [🤗 Hub](https://huggingface.co/models?search=auto-g-nano) |
| **auto-g-nano-1B** | 1B | 256 | 1024 | 16 | 24 | [`auto-g-nano-1b`](https://github.com/williamseemueller/auto-g-nano/tree/auto-g-nano-1b) | [🤗 Hub](https://huggingface.co/models?search=auto-g-nano) |
| **auto-g-nano-1B-CoreML** | 1B | 256 | 1024 | 16 | 24 | [`auto-g-nano-1b-coreml`](https://github.com/williamseemueller/auto-g-nano/tree/auto-g-nano-1b-coreml) | [🤗 Hub](https://huggingface.co/models?search=auto-g-nano) |

**To use any model:** Check out the corresponding branch to find training/inference code and the model weights.

---

## 🏗 Architecture

All models follow the same decoder-only Transformer architecture:
- **Token & Positional Embeddings**
- **Causal Self-Attention**: Multi-head attention with masking to prevent looking into the future.
- **Feed-Forward Networks**: 2-layer MLP with ReLU activation.
- **Layer Normalization**: Applied before each sub-layer (Pre-Norm).
- **Residual Connections**: Crucial for deep network stability.

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/williamseemueller/auto-g-nano.git
   cd auto-g-nano
   ```

2. **Check out your desired model branch:**
   ```bash
   git checkout auto-g-nano-10m  # or auto-g-nano-153M, auto-g-nano-1b, etc.
   ```

3. **Install dependencies and run training/inference:**
   - Each branch contains its own `README.md` with instructions for training, inference, and deployment.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.