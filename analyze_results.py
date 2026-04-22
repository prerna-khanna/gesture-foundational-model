import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

file_path = 'results/final_results/utd_mhad_04_22_2026_15_48/results.csv'
df = pd.read_csv(file_path)

# 1) Total rows, overall accuracy, unique labels
total_rows = len(df)
accuracy = (df['true_label'] == df['predicted_label']).mean()
unique_true = sorted(df['true_label'].unique())
unique_pred = sorted(df['predicted_label'].unique())

print(f"Total rows: {total_rows}")
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Unique True Labels: {unique_true}")
print(f"Unique Pred Labels: {unique_pred}")

# 2) Confusion matrix summary
labels = sorted(list(set(unique_true) | set(unique_pred)))
cm = confusion_matrix(df['true_label'], df['predicted_label'], labels=labels)

print("\nPer-class Support and Accuracy (Recall):")
for i, label in enumerate(labels):
    support = cm[i, :].sum()
    if support > 0:
        correct = cm[i, i]
        acc = correct / support
        print(f"Label {label:2}: Support {support:3}, Accuracy {acc:.4f}")

# 3) Top 10 misclassification pairs
misclassifications = df[df['true_label'] != df['predicted_label']]
top_misclass = misclassifications.groupby(['true_label', 'predicted_label']).size().reset_index(name='count')
top_misclass = top_misclass.sort_values(by='count', ascending=False).head(10)

print("\nTop 10 Misclassification Pairs (True -> Pred):")
for _, row in top_misclass.iterrows():
    print(f"True {row['true_label']:2} -> Pred {row['predicted_label']:2}: {row['count']} occurrences")

# 4) Label range check
expected_range = set(range(27))
actual_range = set(unique_true) | set(unique_pred)
is_subset = actual_range.issubset(expected_range)
print(f"\nLabels in expected range 0-26: {is_subset}")
if not is_subset:
    print(f"Labels found outside 0-26: {actual_range - expected_range}")
