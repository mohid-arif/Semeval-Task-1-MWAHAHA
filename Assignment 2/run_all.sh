#!/bin/bash

# ---------------------------------------
# Run all baselines for MWAHAHA project
# Works in Git Bash / Windows
# ---------------------------------------

# Set Python executable (adjust if needed)
PYTHON=python

# Directories
DATA_DIR="./data"
LOG_DIR="./logs"
PRED_DIR="./predictions"

# Create directories (Windows/Git Bash friendly)
mkdir -p $LOG_DIR
mkdir -p $PRED_DIR


# Train and evaluate GPT-2
echo "Running baseline 3: GPT-2..."
$PYTHON baseline_gpt2.py --data $DATA_DIR/train.tsv --save_model $LOG_DIR/gpt2_model --epochs 3 > $LOG_DIR/gpt2.log 2>&1
$PYTHON eval_gpt2.py --model $LOG_DIR/gpt2_model --data $DATA_DIR/test.tsv --output $PRED_DIR/gpt2_predictions.csv

# Train and evaluate T5 / BART
echo "Running baseline 4: T5/BART..."
$PYTHON baseline_t5.py --data $DATA_DIR/train.tsv --save_model $LOG_DIR/t5_model --epochs 3 > $LOG_DIR/t5.log 2>&1
$PYTHON eval_t5.py --model $LOG_DIR/t5_model --data $DATA_DIR/test.tsv --output $PRED_DIR/t5_predictions.csv

echo "All baselines finished! Predictions saved in $PRED_DIR"
