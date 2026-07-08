import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).resolve().parent
data_file = script_dir / "Results.csv"
output_dir = script_dir / "Dataplots" / "DemographicHistograms"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_file, header=None)

# Respondent rows (listener answers), columns 1-36 are ratings, 37-43 are demographics
respondents = df.iloc[2:87]

# Row 88 holds speakers' actual self-reported sexuality scores (0-10 scale), columns 1-36
actual_scores = pd.to_numeric(df.iloc[88, 1:37], errors="coerce").dropna()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(actual_scores, bins=[x - 0.5 for x in range(12)],
        edgecolor="black", color="steelblue", alpha=0.8)
ax.set_title("Distribution of Speakers' Actual Sexuality Scores")
ax.set_xlabel("Self-Reported Score (0 = Exclusively Straight, 10 = Exclusively Gay)")
ax.set_ylabel("Number of Speakers")
ax.set_xticks(range(11))
fig.savefig(output_dir / "ActualSexualityScores.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: ActualSexualityScores.png")

# Listener demographic questions, columns 37-43
demographics = [
    {"col": 37, "title": "Listener Year of Birth",
     "xlabel": "Year of Birth", "labels": None},
    {"col": 38, "title": "Listener Gender Identity",
     "xlabel": "Gender", "labels": {1: "Male", 2: "Female"}},
    {"col": 39, "title": "Listener Sexual Orientation",
     "xlabel": "Sexual Orientation",
     "labels": {1: "Gay/Lesbian", 2: "Straight", 3: "Bi/Pan", 4: "Other", 5: "Prefer not to say"}},
    {"col": 40, "title": "Did Listener Speak English as a Child",
     "xlabel": "Spoke English as a Child", "labels": {1: "Yes", 2: "No"}},
    {"col": 41, "title": "Listener's Number of Gay Friends",
     "xlabel": "Gay Friends", "labels": {1: "None", 2: "1-2", 3: "3 or more"}},
    {"col": 42, "title": "Listener's Number of Gay Acquaintances",
     "xlabel": "Gay Acquaintances", "labels": {1: "None", 2: "1-2", 3: "3 or more"}},
    {"col": 43, "title": "Listener Religiosity",
     "xlabel": "Religiosity (0 = Atheist, 10 = Very Religious)", "labels": None},
]

for demo in demographics:
    values = pd.to_numeric(respondents.iloc[:, demo["col"]], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(8, 5))

    if demo["labels"]:
        order = sorted(demo["labels"])
        counts = values.value_counts().reindex(order, fill_value=0)
        ax.bar([demo["labels"][k] for k in order], counts.values,
               edgecolor="black", color="steelblue", alpha=0.8)
    else:
        lo, hi = int(values.min()), int(values.max())
        ax.hist(values, bins=[x - 0.5 for x in range(lo, hi + 2)],
                edgecolor="black", color="steelblue", alpha=0.8)

    ax.set_title(demo["title"])
    ax.set_xlabel(demo["xlabel"])
    ax.set_ylabel("Number of Listeners")

    filename = demo["title"].replace("'", "").replace(" ", "_") + ".png"
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")

print("\nDone! All demographic histograms saved.")
