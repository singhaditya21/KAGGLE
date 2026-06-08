# NVIDIA Nemotron Model Reasoning Challenge

A self-contained project workspace for fine-tuning and evaluating NVIDIA's Nemotron-3-Nano-30B model on structured reasoning puzzles.

## Project Structure

```
.
├── configs/            # JSON/YAML configuration files for training parameters
├── data/               # All raw, processed, and external datasets (Git ignored)
│   └── raw/            # Contains train.csv and test.csv
├── src/                # Source code and utility scripts
│   ├── download_data.py
│   ├── experiment_utils.py
│   └── submission_budget.py
├── pyproject.toml      # Dependency declaration managed by uv
└── README.md
```

## Competition Details

* **Task:** Predict the final answer (enclosed in `\boxed{}`) to multi-step reasoning puzzles (logic ciphers, equation transformations, physics calculations, etc.).
* **Target Format:** Submit a **LoRA adapter (rank <= 32)** packaged into a `submission.zip` containing `adapter_config.json` and the weights file, rather than a prediction CSV.
* **Inference Engine:** Evaluated using vLLM on the base `Nemotron-3-Nano-30B` model with internet disabled.

## Recommended Workflows

1. **Local EDA and Rule-Based Solvers:**
   * Explore the puzzle types in `data/raw/train.csv` (e.g. binary ciphers, Roman numerals, unit conversions).
   * For structured tasks (like Roman numerals or standard unit conversions), rule-based parser scripts can achieve 100% accuracy.

2. **Fine-Tuning on Kaggle Kernels:**
   * Since fine-tuning a 30B model requires significant VRAM, run SFT (Supervised Fine-Tuning) using Unsloth, Axolotl, or TRL directly inside a Kaggle Notebook utilizing dual-T4 GPUs or TPU resources.
   * Export the LoRA adapter weights, zip them into `submission.zip`, and submit!
