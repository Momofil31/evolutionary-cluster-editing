
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the main data
data = pd.read_csv("visualization_csv/best_fitness.csv")
updated_name_mapping = {
    "0_cliques_ring": "cliques_ring",
    "1_cliques_binarytree": "cliques_binary_tree",
    "2_quasicliques": "cliques_quasicliques",
    "3_cliques_tree": "cliques_tree_small",
    "4_cliques_tree": "cliques_tree_big",
    "5_cactus": "cactus_small",
    "6_cactus": "cactus_big",
    "7_partition": "partition_small",
    "8_partition": "partition_big",
    "9_sbm": "sbm_A",
    "10_sbm": "sbm_B",
    "11_sbm": "sbm_C",
    "12_sbm": "sbm_D",
}

# Extract the number prefix and order the data
data['order'] = data['input_file'].str.split('_').str[0].astype(int)
grouped_time_elapsed = data.groupby(['order', 'input_file', 'Group'])['time_elapsed'].mean().unstack().reset_index().sort_values('order')
grouped_time_elapsed['input_file'] = grouped_time_elapsed['input_file'].map(updated_name_mapping)

# Plotting
palette_set2 = sns.color_palette("Set2", n_colors=len(grouped_time_elapsed.columns) - 2)
fig, ax = plt.subplots(figsize=(12, 7))

# Define bar positions and width
x = np.arange(len(grouped_time_elapsed['input_file']))
width = 0.25

# Plot bars for each algorithm group
for i, (column, color) in enumerate(zip(grouped_time_elapsed.columns[2:], palette_set2)):
    ax.bar(x + i*width, grouped_time_elapsed[column], width, label=column, color=color)

# Customize the plot
ax.set_xlabel('Test Cases')
ax.set_ylabel('Time Elapsed (s)')
ax.set_title('Time Elapsed by Test Case and Algorithm')
ax.set_xticks(x + width)
ax.set_xticklabels(grouped_time_elapsed['input_file'], rotation=45)
ax.legend()
ax.grid(axis='y')

fig.tight_layout()
plt.savefig("plots/time_elapsed.png", dpi=300, bbox_inches='tight')
plt.show()
