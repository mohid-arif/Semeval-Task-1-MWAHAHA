import pandas as pd
from better_profanity import profanity

# === CONFIGURATION ===
INPUT_CSV = "shortjokes.csv"               # Input file
OUTPUT_CSV = "filtered_jokes.csv"     # File with safe jokes only

# Load default profanity dictionary (includes explicit terms)
profanity.load_censor_words()


def contains_inappropriate(text: str) -> bool:
    """Return True if the joke contains profanity or inappropriate content."""
    if not isinstance(text, str):
        return False
    return profanity.contains_profanity(text)


def main():
    print("Loading CSV...")
    df = pd.read_csv(INPUT_CSV)

    if "Joke" not in df.columns:
        raise ValueError("The CSV must contain a 'Joke' column.")

    print("Filtering inappropriate jokes...")
    df["is_inappropriate"] = df["Joke"].apply(contains_inappropriate)

    # Keep only appropriate jokes
    df_filtered = df[df["is_inappropriate"] == False].drop(columns=["is_inappropriate"])

    print(f"Saving filtered jokes to {OUTPUT_CSV}...")
    df_filtered.to_csv(OUTPUT_CSV, index=False)

    print("Done! Safe jokes file created:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
