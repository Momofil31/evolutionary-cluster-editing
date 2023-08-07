import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils.timed_halloffame import TimedHallOfFame

def save_plots(plot_path, df: pd.DataFrame, hall_of_fame: TimedHallOfFame):
    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.grid(False)

    # Set the style of the plots
    sns.set(style="whitegrid")

    # Plot fitness_min with ax1, use 'skyblue' color
    sns.lineplot(
        data=df,
        x=df.index,
        y="fitness_min",
        ax=ax1,
        color="skyblue",
        label="Fitness Min",
    )

    # Create a second y-axis with the same x-axis
    ax2 = ax1.twinx()
    ax2.grid(False)

    # Plot num_clusters_min with ax2, use 'olive' color
    sns.lineplot(
        data=df,
        x=df.index,
        y="num_clusters_min",
        ax=ax2,
        color="orange",
        label="Num Clusters Min",
    )

    # Assuming `hall_of_fame` is your Hall of Fame object
    # Find the individual with the minimum timestamp
    min_timestamp = min(hall_of_fame.history.values(), key=lambda x: x[1])[1]
    for ind, stats in hall_of_fame.history.items():
        if stats[1] == min_timestamp:
            best_individual = ind
            break
    best_individual_stats = hall_of_fame.get_individual_stats(best_individual)

    if best_individual_stats:
        best_gen, best_fitness, best_timestamp = best_individual_stats
        best_fitness_value = df.loc[best_gen, "fitness_min"]

        # Draw vertical line at the generation of the best individual
        ax1.axvline(x=best_gen, linestyle="--", color="gray")

        # Add a red point at the fitness of the best individual
        ax1.scatter(best_gen, best_fitness, color="skyblue", s=50)

        # Construct the caption
        caption = f"Fitness: {best_fitness}\nGen: {best_gen}\nAfter: {best_timestamp:.3f} seconds"

        # Position the caption at the point of the best individual
        # Adjust the values added/subtracted from best_gen and best_fitness_value for your desired caption position
        ax1.annotate(
            caption,
            xy=(best_gen + 0.2, best_fitness_value + 2),
            xycoords="data",
            textcoords="data",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            bbox=dict(facecolor="skyblue", alpha=0.3),
        )

    # Add a title to the plot
    plt.title("Min Fitness and Num Clusters over Iterations")

    # Create a legend for the plot
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Show the plot
    plt.tight_layout()
    plt.savefig(plot_path)