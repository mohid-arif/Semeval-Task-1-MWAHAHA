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
├── data/                   # Input data files (from CodaBench)
│   ├── dev/                # Development phase inputs
│   └── eval/               # Evaluation phase inputs
│
├── src/                    # Source code
│   ├── subtask_a/          # Text-based humor generation pipeline
│   └── subtask_b/          # Image-based humor caption generation
│
├── outputs/                # Generated joke/caption outputs
│
├── baselines/              # Baseline prompts and outputs
│
├── requirements.txt        # Python dependencies
└── README.md
```

> **Note:** Data files are not included in this repository. Download them from the [CodaBench competition page](https://www.codabench.org/competitions/9719/).

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- pip

### Install Dependencies

```bash
git clone https://github.com/mohid-arif/Semeval-Task-1-MWAHAHA.git
cd Semeval-Task-1-MWAHAHA
pip install -r requirements.txt
```

### API Keys

This project may use LLM APIs. Set the relevant environment variables before running:

```bash
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

---

## Usage

### Subtask A — Text Generation

```bash
python src/subtask_a/generate.py \
  --input data/eval/subtask_a_input.json \
  --output outputs/subtask_a_output.json \
  --lang en
```

### Subtask B — Image Caption Generation

```bash
python src/subtask_b/generate.py \
  --input data/eval/subtask_b_input.json \
  --output outputs/subtask_b_output.json \
  --variant b1   # or b2
```

---

## Approach

Our system explores prompt engineering and fine-tuning strategies to generate novel and contextually appropriate humor.

### Subtask A

- **Prompting strategy:** We use chain-of-thought prompting to guide the model through joke structure (setup → punchline) before generating the final output.
- **Constraint handling:** For word inclusion constraints, we prompt the model to naturally integrate the target words into the joke rather than appending them.
- **Multilingual support:** For Spanish and Chinese, we append language instructions to the English base prompt.

### Subtask B

- **Image understanding:** We extract the first frame of each GIF and pass it to a vision-language model along with a humor-focused system prompt.
- **Caption generation:** The model is instructed to keep captions concise (≤ 20 words) and maximally humorous given the visual context.

---

## Evaluation

Submissions are evaluated through **human preference judgments** in a pairwise "battle" format, where annotators choose the funnier of two outputs generated under the same constraint. Rankings are computed using an **Elo-based leaderboard**, hosted at [thefunnier.com](https://thefunnier.com/leaderboard).

No labeled training data is provided by the organizers — participants are free to use any publicly available data, pre-trained models, or APIs.

---

## Results

| Subtask | System | Elo Score | Rank |
|---|---|---|---|
| A (EN) | Our System | — | — |
| A (ES) | Our System | — | — |
| A (ZH) | Our System | — | — |
| B1 | Our System | — | — |
| B2 | Our System | — | — |

*(Results to be updated after competition evaluation.)*

---

## References

- SemEval 2026 Task 1 — MWAHAHA: [Official Website](https://pln-fing-udelar.github.io/semeval-2026-humor-gen/)
- CodaBench Competition Page: [https://www.codabench.org/competitions/9719/](https://www.codabench.org/competitions/9719/)
- SemEval 2026: [https://semeval.github.io/SemEval2026/](https://semeval.github.io/SemEval2026/)
- Chatbot Arena (Elo inspiration): [https://lmarena.ai/](https://lmarena.ai/)

---

*This repository was created as part of a course/research project participation in SemEval 2026.*
