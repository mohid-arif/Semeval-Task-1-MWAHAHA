import pandas as pd
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

# ============================
# Load and clean your dataset
# ============================

csv_path = "merged_models.csv"   # Path to file

df = pd.read_csv(csv_path, engine="python")

# Fix multi-line prompts like "maybe, village\nJoke:"
df["prompt"] = df["prompt"].astype(str).str.replace("\n", " ").str.strip()

# Fix targets with accidental breaks, stray quotes, etc.
df["target"] = df["target"].astype(str).str.replace("\n", " ").str.strip()

# Replace NaNs with empty strings
for col in ["ngram", "seq2seq", "gpt2", "t5"]:
    df[col] = df[col].fillna("").astype(str)

# ============================
# Evaluation setup
# ============================

bleu = BLEU(effective_order=True)
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def compute_bleu(pred, tgt):
    if not tgt.strip():
        return 0.0
    return bleu.sentence_score(pred, [tgt]).score

def compute_rouge(pred, tgt):
    if not tgt.strip():
        return 0.0
    return scorer.score(tgt, pred)["rougeL"].fmeasure

# ============================
# Evaluate each model output
# ============================

for model in ["ngram", "seq2seq", "gpt2", "t5"]:
    bleu_scores = []
    rouge_scores = []

    for _, row in df.iterrows():
        pred = row[model]
        tgt = row["target"]

        bleu_scores.append(compute_bleu(pred, tgt))
        rouge_scores.append(compute_rouge(pred, tgt))

    df[f"BLEU_{model}"] = bleu_scores
    df[f"ROUGE_{model}"] = rouge_scores

# ============================
# Save final evaluated CSV
# ============================

out_path = "evaluated_results.csv"
df.to_csv(out_path, index=False)

print("Evaluation complete!")
print(f"Saved file: {out_path}")

df.head()
