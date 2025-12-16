import os
import re
import itertools
import pandas as pd
from collections import Counter

# === CONFIGURATION ===
CORPUS_FOLDER = "corpus"      # folder with .txt files
UNIGRAM_OUT = "word_freq.csv"
BIGRAM_OUT = "bigram_freq.csv"


def tokenize_words(text):
    """Extract lowercase alphabetic words."""
    return re.findall(r"[a-zA-Z]+", text.lower())


def split_sentences(text):
    """Split text into sentences using punctuation."""
    # Splits on ., !, ?, but keeps things simple and robust
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def process_file(path, unigram_counter, bigram_counter):
    """Process one text file and update counters using sentence-level co-occurrence."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    sentences = split_sentences(text)

    for sentence in sentences:
        words = tokenize_words(sentence)

        if not words:
            continue

        # Update unigram counts globally
        unigram_counter.update(words)

        # Count co-occurrence pairs **within this sentence only**
        unique_words = sorted(set(words))
        pairs = itertools.combinations(unique_words, 2)

        bigram_counter.update(pairs)


def main():
    print("Scanning corpus folder:", CORPUS_FOLDER)

    unigram_counter = Counter()
    bigram_counter = Counter()

    # Walk through all .txt files
    for root, _, files in os.walk(CORPUS_FOLDER):
        for filename in files:
            if filename.lower().endswith(".txt"):
                filepath = os.path.join(root, filename)
                print("Processing:", filepath)
                process_file(filepath, unigram_counter, bigram_counter)

    # Save unigram frequencies
    print("Saving unigram frequencies...")
    unigram_df = pd.DataFrame(
        [(word, freq) for word, freq in unigram_counter.items()],
        columns=["word", "frequency"]
    )
    unigram_df.to_csv(UNIGRAM_OUT, index=False)

    # Save bigram co-occurrence frequencies
    print("Saving sentence-level bigram frequencies...")
    bigram_df = pd.DataFrame(
        [(w1, w2, freq) for (w1, w2), freq in bigram_counter.items()],
        columns=["word1", "word2", "frequency"]
    )
    bigram_df.to_csv(BIGRAM_OUT, index=False)

    print("Done.")
    print(f"Unigrams saved to: {UNIGRAM_OUT}")
    print(f"Bigrams saved to:  {BIGRAM_OUT}")


if __name__ == "__main__":
    main()
