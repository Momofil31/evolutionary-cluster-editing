import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the three CSV files
ev_df = pd.read_csv("visualization_csv/ev_2_quasicliques.csv")
evls_df = pd.read_csv("visualization_csv/evls_2_quasicliques.csv")
ls_df = pd.read_csv("visualization_csv/ls_2_quasicliques.csv")

# Extract the values from the 'norm_min_x' column
ev_df["norm_fitness"] = ev_df["norm_min_x"].str.extract("\[(.*?)\]").astype(float)
evls_df["norm_fitness"] = evls_df["norm_min_x"].str.extract("\[(.*?)\]").astype(float)
ls_df["norm_fitness"] = ls_df["norm_min_x"].str.extract("\[(.*?)\]").astype(float)

# Set the Seaborn "Set2" palette
palette = sns.color_palette("Set2", 6)

# Colors from Set2 palette based on your request
colors = {
    "ev": palette[2],  # greenish
    "evls": palette[1],  # orange
    "ls": palette[0],  # blue
}

# Setting up the figure and primary axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plotting normalized fitness on the primary y-axis using dashed lines
sns.lineplot(
    x=ev_df["gen"],
    y=ev_df["norm_fitness"],
    ax=ax1,
    label="Evolution",
    color=colors["ev"],
    linestyle="--",
)
sns.lineplot(
    x=evls_df["gen"],
    y=evls_df["norm_fitness"],
    ax=ax1,
    label="Evolution with LS",
    color=colors["evls"],
    linestyle="--",
)
sns.lineplot(
    x=ls_df["iter"],
    y=ls_df["norm_fitness"],
    ax=ax1,
    label="Local Search",
    color=colors["ls"],
    linestyle="--",
)

# Setting up the secondary y-axis
ax2 = ax1.twinx()
# Plotting num_clusters_min on the secondary y-axis using full lines
sns.lineplot(x=ev_df["gen"], y=ev_df["num_clusters_avg"], ax=ax2, color=colors["ev"])
sns.lineplot(
    x=evls_df["gen"], y=evls_df["num_clusters_avg"], ax=ax2, color=colors["evls"]
)
sns.lineplot(x=ls_df["iter"], y=ls_df["num_clusters_min"], ax=ax2, color=colors["ls"])

# Setting the title, labels, and legends
ax1.set_title("Scaled Fitness and Number of Clusters Comparison across Evolution")
ax1.set_xlabel("Generation / Iteration")
ax1.set_ylabel("Normalized Fitness")
ax2.set_ylabel("Avg Number of Clusters")
ax1.legend(loc="upper center")
ax1.grid(True)

plt.tight_layout()
plt.savefig("plots/num_clusters_evolution.png", dpi=300, bbox_inches="tight")
plt.show()
