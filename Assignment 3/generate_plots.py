import matplotlib.pyplot as plt
import pandas as pd
import re

# # --- Data from the provided table ---
# steps = [
#     50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
#     800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400,
#     1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000,
#     2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600,
#     2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000, 3050, 3100, 3150, 3200,
#     3250, 3300, 3350, 3400, 3450, 3500, 3550, 3600
# ]

# loss = [
#     0.816700, 0.464300, 0.458500, 0.448700, 0.446500, 0.451500, 0.450300,
#     0.444200, 0.447200, 0.423300, 0.421400, 0.431100, 0.434300, 0.427300,
#     0.435100, 0.422000, 0.418800, 0.413200, 0.416100, 0.404000, 0.413100,
#     0.406500, 0.411700, 0.401300, 0.380000, 0.374800, 0.387800, 0.379500,
#     0.383100, 0.382100, 0.390200, 0.389400, 0.373600, 0.366900, 0.383100,
#     0.378700, 0.362000, 0.378700, 0.367700, 0.366100, 0.378800, 0.365000,
#     0.375700, 0.369200, 0.375000, 0.366900, 0.364300, 0.370200, 0.352200,
#     0.351500, 0.351300, 0.350600, 0.353200, 0.349900, 0.352800, 0.347500,
#     0.346200, 0.349000, 0.349000, 0.350900, 0.346600, 0.349200, 0.351800,
#     0.351900, 0.337900, 0.353300, 0.354300, 0.346400, 0.341100, 0.352800,
#     0.339700, 0.354000
# ]

# # --- Plotting Configuration ---
# plt.figure(figsize=(12, 6))

# # Plotting the data as a line with markers for each data point
# plt.plot(steps, loss, marker='o', markersize=4, linestyle='-', linewidth=1.5, color='#1f77b4', label='Training Loss')

# # Adding titles and labels
# plt.title('GPT-2 Training Loss Curve', fontsize=16, fontweight='bold')
# plt.xlabel('Training Steps', fontsize=12)
# plt.ylabel('Loss', fontsize=12)

# # Adding a grid for easier readability
# plt.grid(True, which='both', linestyle='--', alpha=0.6)

# # Adding a legend
# plt.legend()

# # Ensuring the layout is neat
# plt.tight_layout()

# # Display the plot
# plt.show()

# # Example TSV: 'joke_text' column
# jokes_df = pd.read_csv("test_predictions.tsv", sep="\t")

# # Compute length
# jokes_df['length'] = jokes_df['generated_joke'].str.len()

# plt.figure(figsize=(6,4))
# plt.hist(jokes_df['length'], bins=30, color='purple', edgecolor='black')
# plt.xlabel("Joke Length (characters)")
# plt.ylabel("Frequency")
# plt.title("Distribution of Generated Joke Lengths")
# plt.grid(axis='y', alpha=0.75)
# plt.tight_layout()
# plt.savefig("joke_length_distribution.pdf")
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ---------------------------
# Load generated jokes
# ---------------------------
jokes_df = pd.read_csv("test_predictions.tsv", sep="\t")

# ---------------------------
# Define advanced sarcasm heuristic
# ---------------------------
sarcasm_keywords = [
    "yeah, right", "obviously", "sure", "as if", "totally", "surely", "not really"
]
negation_phrases = ["not really", "as if", "yeah right"]
emoji_cues = ["🙃", "😏", "😒"]

def sarcasm_score(text):
    text_lower = str(text).lower()
    score = 0
    
    # Strong indicator
    if "/s" in text_lower:
        score += 3
    
    # Keywords
    for kw in sarcasm_keywords:
        if kw in text_lower:
            score += 1.5
    
    # Negation phrases
    for phrase in negation_phrases:
        if phrase in text_lower:
            score += 2
    
    # Emoji cues
    for e in emoji_cues:
        if e in text_lower:
            score += 0.5
    
    # Punctuation at the end (optional)
    if text_lower.strip().endswith("!"):
        score += 0.3
    
    # Normalize to 1.0
    max_score = 3 + len(sarcasm_keywords)*1.5 + len(negation_phrases)*2 + len(emoji_cues)*0.5 + 0.3
    return min(score / max_score, 1.0)

# ---------------------------
# Compute sarcasm for each joke
# ---------------------------
jokes_df['sarcasm_score'] = jokes_df['generated_joke'].apply(sarcasm_score)

# ---------------------------
# Save results
# ---------------------------
jokes_df.to_csv("generated_jokes_sarcasm.csv", index=False)

# ---------------------------
# Plot KDE curve for sarcasm scores
# ---------------------------
plt.figure(figsize=(6,4))
sns.kdeplot(jokes_df['sarcasm_score'], fill=True, color='green', bw_adjust=0.3)
plt.xlabel("Sarcasm Score (0 = Low, 1 = High)")
plt.ylabel("Density")
plt.title("Enhanced Sarcasm Quality Distribution (Heuristic)")
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig("sarcasm_quality_curve_enhanced.pdf")
plt.show()

print("Enhanced sarcasm scores computed and KDE curve saved as sarcasm_quality_curve_enhanced.pdf")
