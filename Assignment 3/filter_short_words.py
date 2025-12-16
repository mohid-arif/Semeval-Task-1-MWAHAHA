import csv

input_file = "word_pair_jokes.csv"
output_file = "cleane_word_pair_jokes.csv"

def clean_word_pair(word_pair):
    words = word_pair.split()
    cleaned_words = [w for w in words if len(w) > 2]
    return " ".join(cleaned_words)

with open(input_file, newline="", encoding="utf-8") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames

    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        row["word_pair"] = clean_word_pair(row["word_pair"])
        writer.writerow(row)

print("Cleaning complete. Output saved to:", output_file)
