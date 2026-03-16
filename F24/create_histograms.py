import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the CSV
df = pd.read_csv(r"C:\Users\jinfa\Desktop\GayStudy\F24\Results.csv", header=None)

# Speaker names are in row 0, columns 1-36
speakers = [str(s).strip() for s in df.iloc[0, 1:37]]

# Listener ratings are in rows 2-87 (0-indexed), columns 1-36
ratings_df = df.iloc[2:88, 1:37]

# Converted self-reported scores (row 90, 0-indexed) - on the same 1-5 scale as listener ratings
converted_scores = df.iloc[90, 1:37]

# Actual self-reported scores (row 89, 0-indexed) - on 0-10 scale
actual_scores = df.iloc[89, 1:37]

output_dir = r"C:\Users\jinfa\Desktop\GayStudy\F24\Dataplots\Histograms"

for i, speaker in enumerate(speakers):
    col_ratings = pd.to_numeric(ratings_df.iloc[:, i], errors="coerce").dropna()
    converted_score = pd.to_numeric(converted_scores.iloc[i], errors="coerce")
    actual_score = pd.to_numeric(actual_scores.iloc[i], errors="coerce")

    fig, ax = plt.subplots(figsize=(7, 5))

    # Histogram of listener ratings (1-5 scale), no gaps between bars
    ax.hist(col_ratings, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            edgecolor="black", color="steelblue", alpha=0.7)

    # Vertical red line at converted self-reported score
    if not np.isnan(converted_score):
        ax.axvline(x=converted_score, color="red", linewidth=2, linestyle="--",
                   label=f"Self-reported: {actual_score:.0f}/10 (converted: {converted_score:.1f})")

    ax.set_xlabel("Listener Rating (1-5)")
    ax.set_ylabel("Count")
    ax.set_title(f"Listener Ratings for {speaker}")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend()

    filename = speaker.replace(".", "").replace(" ", "_") + ".png"
    fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")

print("\nDone! All 36 histograms saved.")
