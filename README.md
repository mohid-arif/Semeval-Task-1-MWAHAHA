# SemEval-Task-1-MWAHAHA

Baseline Pipeline Implementation for **SemEval 2026 Task 1: MWAHAHA**  
(Models Write Automatic Humor And Humans Annotate)

---

## 📌 About the Task

**SemEval 2026 Task 1 — MWAHAHA** is a shared task in computational humor generation that challenges participants to build systems capable of generating genuinely humorous content and evaluating it with human annotations.  
The task explores the frontier of humor generation by requiring models to produce creative, funny text under defined constraints, pushing beyond standard language modeling towards **computational creativity**. :contentReference[oaicite:1]{index=1}

The task includes multiple subtasks and languages (e.g., English, Spanish, Chinese), focusing on:
* **Text-based humor generation**
* **Humorous captioning for images and GIFs** (if applicable)

Participants are evaluated based on how well their models produce humorous output that aligns with human judgments. :contentReference[oaicite:2]{index=2}

---

## 🧠 Repository Overview

This repository provides a **baseline pipeline** for participating in SemEval 2026 Task 1: MWAHAHA.  
It includes:
* Core scripts to preprocess data
* Model training and evaluation pipelines
* Examples of generating humor for task submission
* Utility scripts and configuration files

The pipeline is intended as a starting point for experimentation and submission in the shared task leaderboard.

---

## 🚀 Getting Started

### 📦 Requirements

Make sure you have the following installed:

* Python 3.8+
* Relevant NLP libraries (e.g., `transformers`, `torch`)
* Any dataset download scripts (see next section)

Install necessary dependencies:

```bash
pip install -r requirements.txt
