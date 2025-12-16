import pandas as pd
import itertools
import re
import math
from nltk.corpus import wordnet as wn, stopwords
import nltk

# Download required NLTK data
nltk.download("wordnet")
nltk.download("stopwords")

# === REQUIRED INPUT FILES ===
JOKES_CSV = "filtered_jokes.csv"              # Input jokes: ID, Joke
UNIGRAM_FREQ = "word_freq.csv"       # Used for frequency data (NOT dictionary)
BIGRAM_FREQ = "bigram_freq.csv"

OUTPUT_CSV = "word_pair_jokes.csv"

# Load stopwords
STOPWORDS = set(stopwords.words("english"))


def tokenize(text):
    """Extract lowercase alphabetic words."""
    return re.findall(r"[a-zA-Z]+", text.lower())


def is_in_dictionary(word):
    """Check if a word exists in WordNet (official dictionary)."""
    return len(wn.synsets(word)) > 0


def load_frequencies():
    """Load unigram+bigram frequency dictionaries safely."""
    unigram_df = pd.read_csv(UNIGRAM_FREQ)
    unigram_freq = dict(zip(unigram_df["word"].astype(str).str.lower(),
                            unigram_df["frequency"]))

    bigram_df = pd.read_csv(BIGRAM_FREQ)

    bigram_freq = {}

    for _, row in bigram_df.iterrows():
        w1 = row.get("word1")
        w2 = row.get("word2")

        # Skip missing (NaN) values
        if not isinstance(w1, str) or not isinstance(w2, str):
            continue

        w1 = w1.lower()
        w2 = w2.lower()

        # Skip anything not purely alphabetic
        if not w1.isalpha() or not w2.isalpha():
            continue

        freq = row.get("frequency", 0)
        if pd.isna(freq):
            continue

        bigram_freq[(w1, w2)] = freq
        bigram_freq[(w2, w1)] = freq  # treat as unordered

    return unigram_freq, bigram_freq


def compute_pmi(w1, w2, unigram_freq, bigram_freq, total_unigrams, total_bigrams):
    """Calculate PMI; high PMI = strong association. Low = rare."""
    f1 = unigram_freq.get(w1, 0)
    f2 = unigram_freq.get(w2, 0)
    f12 = bigram_freq.get((w1, w2), 0)

    if f1 == 0 or f2 == 0:
        return None  # missing from frequency table
    if f12 == 0:
        return -999  # extremely rare co-occurrence

    # PMI formula
    p1 = f1 / total_unigrams
    p2 = f2 / total_unigrams
    p12 = f12 / total_bigrams

    return math.log(p12 / (p1 * p2), 2)


def main():
    print("Loading frequencies...")
    unigram_freq, bigram_freq = load_frequencies()

    total_unigrams = sum(unigram_freq.values())
    total_bigrams = sum(bigram_freq.values())

    print("Loading jokes...")
    df = pd.read_csv(JOKES_CSV)

    results = []

    print("Processing jokes...")
    for _, row in df.iterrows():
        joke_id = row["ID"]
        joke = str(row["Joke"])
        words = tokenize(joke)

        # Filter using WordNet + remove stopwords
        dict_words = [
            w for w in words
            if is_in_dictionary(w) and w not in STOPWORDS
        ]

        # Unique word pairs
        pairs = itertools.combinations(sorted(set(dict_words)), 2)

        for w1, w2 in pairs:

            # Skip if frequencies not available
            if w1 not in unigram_freq or w2 not in unigram_freq:
                continue

            pmi = compute_pmi(w1, w2, unigram_freq, bigram_freq,
                              total_unigrams, total_bigrams)

            if pmi is None:
                continue

            # Threshold for "rarely used together"
            if pmi < -3:
                results.append([joke_id, f"{w1} {w2}", joke])

    print("Saving results to", OUTPUT_CSV)
    out_df = pd.DataFrame(results, columns=["ID", "word_pair", "Joke"])
    out_df.to_csv(OUTPUT_CSV, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
