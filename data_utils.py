import pandas as pd
import torch
from torch.utils.data import Dataset

# data_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_semeval_dataset(path):
    """
    Load the SEMEVAL humor dataset.
    Expects columns: word1, word2, headline (id optional)
    Returns a pandas DataFrame with columns: 'id', 'prompt', 'target'
    """
    import os
    ext = os.path.splitext(path)[1]
    if ext in [".tsv", ".txt"]:
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    
    df = df.fillna("-")

    # Generate 'prompt' column
    df['prompt'] = df.apply(
        lambda row: row['headline'] if row.get('headline', '-') != "-" else f"{row.get('word1','')} {row.get('word2','')}", axis=1
    )
    # Generate 'target' column
    df['target'] = df.apply(lambda row: row['headline'] if row.get('headline', '-') != "-" else "-", axis=1)

    # Add 'id' if missing
    if 'id' not in df.columns:
        df.insert(0, 'id', range(len(df)))

    return df[['id','prompt','target']]


def train_val_test_split(df, val_size=0.1, test_size=0.1, random_state=42):
    """
    Split dataframe into train, val, test
    """
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=random_state)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def save_preds(path, ids, prompts, preds, refs=None):
    """
    Save predictions to CSV
    """
    df = pd.DataFrame({"id": ids, "prompt": prompts, "prediction": preds})
    if refs is not None:
        df["reference"] = refs
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------
# 1. Unified prompt formatting
# ---------------------------------------------------------
def build_prompt(row):
    """
    Convert each row into a unified generation prompt.
    If headline exists → Headline prompt
    If word1 & word2 exist → Word-pair prompt
    """
    if row["headline"] != "-" and isinstance(row["headline"], str):
        # Headline-based humor generation
        return f"Headline: {row['headline']}\nJoke:"
    else:
        # Word-pair humor generation
        return f"Words: {row['word1']}, {row['word2']}\nJoke:"


# ---------------------------------------------------------
# 2. Dataset for N-gram, LSTM, GPT-2, T5/BART
# ---------------------------------------------------------
class HumorDataset(Dataset):
    def __init__(self, path, tokenizer=None, max_length=128, model_type="gpt2"):
        self.df = pd.read_csv(path, sep="\t")
        self.df.fillna("-", inplace=True)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type

        # Build textual prompts
        self.df["prompt"] = self.df.apply(build_prompt, axis=1)

        # For training generative models, we need: prompt → target_text
        # BUT since baseline models need training jokes, you must have a "joke" column in train set.
        if "joke" in self.df.columns:
            self.df["target"] = self.df["joke"]
        else:
            # For test set (no ground truth)
            self.df["target"] = ""

        # Full training string
        self.df["input_text"] = self.df["prompt"] + " " + self.df["target"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt = self.df.iloc[idx]["prompt"]
        target = self.df.iloc[idx]["target"]
        full_text = self.df.iloc[idx]["input_text"]

        if self.tokenizer:
            if self.model_type in ["gpt2", "gpt-neo"]:
                # Causal LM: Single sequence with labels shifted internally
                enc = self.tokenizer(
                    full_text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                return {
                    "input_ids": enc["input_ids"].squeeze(),
                    "attention_mask": enc["attention_mask"].squeeze(),
                    "labels": enc["input_ids"].squeeze(),
                }

            else:
                # Seq2Seq: prompt → target
                enc_inp = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                enc_tgt = self.tokenizer(
                    target,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                return {
                    "input_ids": enc_inp["input_ids"].squeeze(),
                    "attention_mask": enc_inp["attention_mask"].squeeze(),
                    "labels": enc_tgt["input_ids"].squeeze(),
                }

        # For non-tokenizer baselines (e.g., N-gram LM)
        return {
            "prompt": prompt,
            "target": target,
            "input_text": full_text,
        }
