import csv
import random

# Input files
WORDPAIR_CSV = "trimmed_word_pair_jokes.csv"
HEADLINE_CSV = "headline_jokes_top_level_sarcasm.csv"
OUTPUT_TSV = "cleaned_combined_jokes.tsv"

rows = []

# ---- Process word-pair file ----
with open(WORDPAIR_CSV, newline="", encoding="utf-8") as wp_f:
    reader = csv.DictReader(wp_f)
    for row in reader:
        word_pair = row["word_pair"].strip()
        joke = row["Joke"].strip()

        parts = word_pair.split()
        word1 = parts[0] if len(parts) > 0 else "-"
        word2 = parts[1] if len(parts) > 1 else "-"

        rows.append([word1, word2, "-", joke])

# ---- Process headline file ----
with open(HEADLINE_CSV, newline="", encoding="utf-8") as hl_f:
    reader = csv.DictReader(hl_f)
    for row in reader:
        headline = row["headline"].strip()
        joke = row["joke"].strip()

        rows.append(["-", "-", headline, joke])

# ---- Shuffle all rows together ----
random.shuffle(rows)

# ---- Write TSV ----
with open(OUTPUT_TSV, "w", newline="", encoding="utf-8") as out_f:
    writer = csv.writer(out_f, delimiter="\t")
    writer.writerow(["word1", "word2", "headline", "joke"])
    writer.writerows(rows)

print("Shuffled combined TSV written to:", OUTPUT_TSV)
