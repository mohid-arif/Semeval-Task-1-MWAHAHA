import csv
import json
import re

# ==============================
# CONFIG
# ==============================
INPUT_TSV = "cleaned_combined_jokes.tsv"
OUTPUT_JSONL = "final_train.jsonl"

# ==============================
# HELPERS
# ==============================
def clean_text(text):
    """Normalize whitespace and remove wrapping quotes"""
    if text is None:
        return ""
    text = text.strip()

    # Remove wrapping quotes
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ==============================
# CONVERT
# ==============================
with open(INPUT_TSV, newline='', encoding="utf-8") as infile, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile, delimiter="\t")

    for row_num, row in enumerate(reader, start=1):
        word1 = clean_text(row.get("word1", ""))
        word2 = clean_text(row.get("word2", ""))
        headline = clean_text(row.get("headline", ""))
        joke = clean_text(row.get("joke", ""))

        # Skip empty jokes
        if not joke:
            continue

        # ==============================
        # Determine prompt type
        # ==============================
        if headline and headline != "-":
            prompt = f"Headline: {headline}\nJoke:"
        elif word1 and word2 and word1 != "-" and word2 != "-":
            prompt = f"Words: {word1}, {word2}\nJoke:"
        else:
            # Skip rows that don't fit either type
            continue

        record = {
            "prompt": prompt,
            "completion": " " + joke
        }

        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Conversion complete → {OUTPUT_JSONL}")
