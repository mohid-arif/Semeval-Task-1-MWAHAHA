import csv
from collections import defaultdict

INPUT_FILE = "cleaned_word_pair_jokes.csv"
OUTPUT_FILE = "trimmed_word_pair_jokes.csv"

MAX_WORD_PAIRS_PER_JOKE = 3
MAX_TOTAL_ROWS = 15000

def has_exactly_two_words(word_pair):
    return len(word_pair.strip().split()) == 2

joke_counts = defaultdict(int)
kept_rows = []

with open(INPUT_FILE, newline="", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)

    for row in reader:
        # Stop if we reached the global row limit
        if len(kept_rows) >= MAX_TOTAL_ROWS:
            break

        word_pair = row["word_pair"]
        joke = row["Joke"]

        # Rule 1: word_pair must have exactly 2 words
        if not has_exactly_two_words(word_pair):
            continue

        # Rule 2: max 3 word pairs per joke
        if joke_counts[joke] >= MAX_WORD_PAIRS_PER_JOKE:
            continue

        kept_rows.append(row)
        joke_counts[joke] += 1

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(kept_rows)

print(f"Done. Kept {len(kept_rows)} rows.")
