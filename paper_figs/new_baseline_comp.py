import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# Data
t1b1 = [0.62, 0.60, 0.59, 0.86, 0.62, 0.76, 0.63, 0.64, 0.51, 0.52, 0.61, 0.65, 
        0.60, 0.70, 0.67, 0.72, 0.75, 0.78, 0.69]
t1b2 = [0.78, 0.85, 0.9, 0.82, 0.88]

t2b1 = [0.57, 0.47, 0.33, 0.22, 0.13, 0.22]
t2b2 = [0.8, 0.9, 0.85, 0.8, 0.81, 0.88]

t3b1 = [0.66, 0.61, 0.6, 0.55]
t3b2 = [0.8, 0.77, 0.91, 0.84, 0.86]

# Calculate means and standard errors
def calculate_stats(data):
    mean = np.mean(data)
    se = stats.sem(data)  # Standard Error of the Mean
    return mean, se

# Calculate statistics
means_b1 = [calculate_stats(t1b1)[0], calculate_stats(t2b1)[0], calculate_stats(t3b1)[0]]
errors_b1 = [calculate_stats(t1b1)[1], calculate_stats(t2b1)[1], calculate_stats(t3b1)[1]]

means_b2 = [calculate_stats(t1b2)[0], calculate_stats(t2b2)[0], calculate_stats(t3b2)[0]]
errors_b2 = [calculate_stats(t1b2)[1], calculate_stats(t2b2)[1], calculate_stats(t3b2)[1]]

# Colors
colors_b1 = ['#ebac8b', '#e56a6f', '#b56577']
color_b2 = 'green'  # Green for all second bars

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(5.5, 2.5))

# Set the width of the bars
bar_width = 0.2
index = np.arange(3)  # Three x-ticks

# Create the bars
bar1 = ax.bar(index - bar_width/2, means_b1, bar_width, 
              yerr=errors_b1, capsize=3, 
              color=colors_b1, edgecolor='black', linewidth=1)

bar2 = ax.bar(index + bar_width/2, means_b2, bar_width, 
              yerr=errors_b2, capsize=3, 
              color=color_b2, edgecolor='black', linewidth=1)

# Customize the plot
ax.set_ylim(0, 1.05)
ax.set_ylabel('Accuracy')
ax.set_xticks(index)
ax.set_xticklabels(['Sightes User\nHand Gesture', 'Sighted User\nEarbud Gesture', 'Blind User\nHand Gesture'])
# add grid for better readability and send it to the back
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

# Create custom legend
patches = []
patches.append(mpatches.Patch(color=colors_b1[0], label='Serendipity [59]'))
patches.append(mpatches.Patch(color=colors_b1[1], label='EarBender [4]'))
patches.append(mpatches.Patch(color=colors_b1[2], label='[27]'))
patches.append(mpatches.Patch(color=color_b2, label='$\it{GestureLens}$'))
ax.legend(handles=patches, ncol = 4, fontsize=9, bbox_to_anchor=(0.5, 1.15), loc='center')
plt.tight_layout()
plt.savefig('paper_figs/new_baseline_comp.pdf')