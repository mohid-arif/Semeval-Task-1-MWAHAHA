import json

def analyze_jokes(jsonl_path):
    headline_count = 0
    wordpair_count = 0

    headline_lengths = []
    wordpair_lengths = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            prompt = data.get("prompt", "")
            joke = data.get("completion", "").strip()

            # joke length = number of words
            joke_length = len(joke.split())

            if "Headline:" in prompt:
                headline_count += 1
                headline_lengths.append(joke_length)

            elif "Words:" in prompt:
                wordpair_count += 1
                wordpair_lengths.append(joke_length)

    avg_headline_len = (
        sum(headline_lengths) / len(headline_lengths)
        if headline_lengths else 0
    )

    avg_wordpair_len = (
        sum(wordpair_lengths) / len(wordpair_lengths)
        if wordpair_lengths else 0
    )

    print("Total headline jokes:", headline_count)
    print("Total word pair jokes:", wordpair_count)
    print("Average joke length (headline jokes):", round(avg_headline_len, 2))
    print("Average joke length (word pair jokes):", round(avg_wordpair_len, 2))


# Example usage
if __name__ == "__main__":
    analyze_jokes("final_train.jsonl")
