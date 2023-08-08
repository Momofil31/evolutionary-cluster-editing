
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the main data and bounds data
data = pd.read_csv("visualization_csv/best_fitness.csv")
bounds_data = pd.read_csv("data/bounds/bounds.csv")

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

# Merge the datasets based on the input_file names
merged_data = data.merge(bounds_data, on='input_file', how='left')

# Scale the best_fitness values based on the bounds
merged_data['scaled_fitness'] = (merged_data['best_fitness'] - merged_data['l_bound']) / (merged_data['u_bound'] - merged_data['l_bound'])

# Extract the number prefix and order the data
merged_data['order'] = merged_data['input_file'].str.split('_').str[0].astype(int)
grouped_scaled_fitness = merged_data.groupby(['order', 'input_file', 'Group'])['scaled_fitness'].mean().unstack().reset_index().sort_values('order')
grouped_scaled_fitness['input_file'] = grouped_scaled_fitness['input_file'].map(updated_name_mapping)

# Plotting
palette_set2 = sns.color_palette("Set2", n_colors=len(grouped_scaled_fitness.columns) - 2)
fig, ax = plt.subplots(figsize=(12, 7))

# Define bar positions and width
x = np.arange(len(grouped_scaled_fitness['input_file']))
width = 0.25

# Plot bars for each algorithm group
for i, (column, color) in enumerate(zip(grouped_scaled_fitness.columns[2:], palette_set2)):
    ax.bar(x + i*width, grouped_scaled_fitness[column], width, label=column, color=color)

# Add a dotted line at y=1.0
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1)

# Customize the plot
ax.set_xlabel('Test Cases')
ax.set_ylabel('Scaled Fitness')
ax.set_title('Scaled Fitness by Test Case and Algorithm')
ax.set_xticks(x + width)
ax.set_xticklabels(grouped_scaled_fitness['input_file'], rotation=45)
ax.legend()
ax.grid(axis='y')
y_axis_upper_limit = grouped_scaled_fitness.iloc[:, 2:].max().max() + 0.1
ax.set_ylim(0, y_axis_upper_limit)

fig.tight_layout()
plt.savefig("plots/scaled_fitness.png", dpi=300, bbox_inches='tight')
plt.show()
