import opensmile
import pandas as pd
import os

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

audio_dir = 'clean_wavs'
rows = []

for filename in os.listdir(audio_dir):
    if filename.endswith('.wav'):
        parts = filename.replace('.wav', '').split('_')
        initials = ''.join([p[0] for p in parts])
        filepath = os.path.join(audio_dir, filename)
        features = smile.process_file(filepath)
        features.insert(0, 'ID', initials)
        rows.append(features)

feature_matrix = pd.concat(rows, ignore_index=True)
feature_matrix.to_csv('features.csv', index=False)
print(feature_matrix.shape)  # should be (50, 89) — 88 features + ID

# meta = pd.read_csv('Master_Spreadsheet.csv')

# merged = feature_matrix.merge(
#     meta[['Initials', 'Kinsey Scale (1-5)', 'Self-Described Sexual Orientation', 'Gender ID', 'First Language']],
#     left_on='ID',
#     right_on='Initials'
# )

# merged.to_csv('features_with_labels.csv', index=False)