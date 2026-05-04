# SemEval 2026 Task 1 — MWAHAHA: Models Write Automatic Humor And Humans Annotate

> *Can computers be funny?*

This repository contains our system submission for **SemEval 2026 Task 1 — MWAHAHA**, the first shared task dedicated to **Computational Humor Generation**. The task challenges participants to build systems that generate genuinely humorous content under various constraints, pushing models beyond memorization toward true creative humor.

---

## 📌 Table of Contents

- [Task Overview](#task-overview)
- [Subtasks](#subtasks)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Approach](#approach)
- [Evaluation](#evaluation)
- [Results](#results)
- [References](#references)

---

## Task Overview

MWAHAHA is **SemEval 2026's Task 1**, organized by researchers from Universidad de la República, the University of Michigan, and the University of Edinburgh. Unlike prior humor tasks that focus on *understanding* humor, MWAHAHA targets *generation* — a significantly harder and less explored frontier.

Participating systems must generate humorous content under carefully designed constraints to:
- Prevent simple retrieval of existing jokes from the web
- Encourage novelty and genuine creativity
- Ensure fair comparison across systems

Evaluation is based entirely on **human preference judgments** in a pairwise arena-style setup, ranked using an Elo-based leaderboard.

Official competition page: [https://pln-fing-udelar.github.io/semeval-2026-humor-gen/](https://pln-fing-udelar.github.io/semeval-2026-humor-gen/)

---

## Subtasks

### Subtask A — Text-Based Humor Generation

Given a set of text-based constraints, generate a joke. Supported languages: **English, Spanish, and Chinese**.

Each generated joke must satisfy one of the following constraints:

| Constraint | Description |
|---|---|
| **Word Inclusion** | The joke must contain two specific rare words |
| **News Headline** | The joke must relate to (or be inspired by) a given news article headline |

### Subtask B — Multimodal Humor Generation with Images *(English only)*

Given a GIF image, generate a humorous caption (max 20 words):

- **Subtask B1:** Caption inspired only by the GIF image
- **Subtask B2:** Caption that completes a given text prompt using the GIF as context

---

## Repository Structure

```
Semeval-Task-1-MWAHAHA/
│
├── Assignment3.ipynb       # Main notebook: GPT-2 fine-tuning & joke generation pipeline
├── .gitattributes          # Git line-ending configuration
└── README.md
```

The entire pipeline — data loading, model fine-tuning, and inference — lives in `Assignment3.ipynb`, designed to run on **Google Colab** with a GPU runtime (T4).

> **Note:** The training data (`output.jsonl`) and test file are not included in this repository. They must be uploaded manually when running the notebook in Colab.

---

## Setup & Installation

The notebook runs on **Google Colab**. No local installation is required.

1. Open the notebook directly in Colab:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohid-arif/Semeval-Task-1-MWAHAHA/blob/main/Assignment3.ipynb)

2. Set the runtime to **GPU** (T4 recommended): `Runtime → Change runtime type → T4 GPU`

3. Run all cells in order. When prompted, upload your `output.jsonl` training file and test CSV using the Colab file upload widget.

### Dependencies

The notebook installs all required packages automatically:

```bash
pip install transformers datasets accelerate sentencepiece
```

**Key libraries:**

| Library | Version | Purpose |
|---|---|---|
| `transformers` | 4.57.3+ | GPT-2 model & tokenizer |
| `datasets` | latest | Data loading from JSONL |
| `accelerate` | latest | Training acceleration |
| `pandas` | latest | Test data processing |
| `tqdm` | latest | Progress bars |

---

## Usage

All steps are contained within `Assignment3.ipynb`. The workflow proceeds as follows:

**1. Upload training data** — use the Colab file upload cell to provide `output.jsonl`

**2. Fine-tune the model** — training runs automatically with the configured hyperparameters

**3. Generate jokes** — the inference cell loads your fine-tuned model and generates jokes from test prompts

**4. Download the model** — the trained model is zipped and downloaded as `gpt2-jokes-final.zip`

---

## Approach

Our system fine-tunes **GPT-2** on a jokes dataset formatted as prompt-completion pairs, then uses the fine-tuned model to generate novel jokes at inference time.

### Model

- **Base model:** `gpt2` (OpenAI GPT-2, loaded via HuggingFace Transformers)
- **Task:** Causal language modelling fine-tuned on joke prompt-completion pairs
- **Tokenizer:** `GPT2Tokenizer` with `pad_token` set to `eos_token`

### Training

The model is fine-tuned using HuggingFace's `Trainer` API with the following configuration:

```python
TrainingArguments(
    output_dir="./gpt2-jokes",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # effective batch size = 16
    learning_rate=5e-5,
    num_train_epochs=3,
)
```

- **Max sequence length:** 256 tokens
- **Train/validation split:** 95% / 5% (seed=42)
- **Hardware:** Google Colab T4 GPU

### Inference

After training, jokes are generated using HuggingFace's `pipeline` API:

```python
generator = pipeline("text-generation", model="gpt2-jokes-final", device=0)
prompt = "Headline: Government promises reform\nJoke:"
```

The model completes the prompt to produce a joke. Test inputs are read from a CSV file and predictions are written to an output file.

---

## Evaluation

Submissions are evaluated through **human preference judgments** in a pairwise "battle" format, where annotators choose the funnier of two outputs generated under the same constraint. Rankings are computed using an **Elo-based leaderboard**, hosted at [thefunnier.com](https://thefunnier.com/leaderboard).

No labeled training data is provided by the organizers — participants are free to use any publicly available data, pre-trained models, or APIs.

---

## References

- SemEval 2026 Task 1 — MWAHAHA: [Official Website](https://pln-fing-udelar.github.io/semeval-2026-humor-gen/)
- CodaBench Competition Page: [https://www.codabench.org/competitions/9719/](https://www.codabench.org/competitions/9719/)
- SemEval 2026: [https://semeval.github.io/SemEval2026/](https://semeval.github.io/SemEval2026/)
- Chatbot Arena (Elo inspiration): [https://lmarena.ai/](https://lmarena.ai/)

---

*This repository was created as part of a research project participation in SemEval 2026.*
