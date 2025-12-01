# MWAHAHA: Humor Generation Baselines

This repository contains baseline implementations for SemEval-2026 Task A: **Computational Humor Generation**. It includes four models (N-gram, Seq2Seq, GPT-2, T5), preprocessing utilities, evaluation scripts, and generated predictions.

## Repository Structure

  
  
baseline_gpt2.py # GPT-2 fine-tuning baseline  
baseline_ngram.py # N-gram LM baseline  
baseline_seq2seq.py # LSTM Seq2Seq baseline  
baseline_t5.py # T5 text-to-text baseline  
data_utils.py # Preprocessing + dataset loading  
evaluate_models.py # BLEU/ROUGE evaluation  
evaluated_results.csv # Final evaluation metrics  
merged_models.csv # Combined predictions  
requirements.txt  
run_all.sh # End-to-end pipeline  
data/train.tsv # Training data  
outputs/ # Model predictions  


## Usage

### Install dependencies:

pip install -r requirements.txt


### Run all baselines and evaluations:

bash run_all.sh


### Run an individual model:

python baseline_ngram.py  
python baseline_seq2seq.py  
python baseline_gpt2.py  
python baseline_t5.py  
  

### Evaluate predictions:

python evaluate_models.py  

### Predictions are stored in:

outputs/  
├── GPT2_preds.csv  
├── ngram_preds.csv  
├── seq2seq_preds.csv  
└── T5_preds.csv  
  

## Evaluation summaries:

evaluated_results.csv — BLEU/ROUGE scores  
  
merged_models.csv — unified predictions from all models  
