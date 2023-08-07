import random
from typing import Any, Dict, List, Optional, Tuple
from operator import itemgetter

import os
import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf
import networkx as nx
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from operators.evaluation import evaluate
from src.utils import pylogger
import src.utils.utils as utils
from src.utils.plots import save_plots
from src.modules.graph_module import GraphModule
from src.modules.local_search_module import LocalSearchModule
from src.utils.logging_utils import log_hyperparameters
from src.utils.instantiators import instantiate_loggers


from src.utils.utils import print_config_tree

log = pylogger.get_pylogger(__name__)


@utils.task_wrapper
def run(
    cfg: DictConfig,
    input_file: str = "0_cliques_ring.txt",
    bounds_file="0_cliques_ring.txt",
    logger=None,
):
    random.seed(64)

    log.info(f"Instantiating <{cfg.data._target_}>")
    graph_module: GraphModule = hydra.utils.instantiate(cfg.data)

    # Load Graph data
    graph_module.load_graph(input_file, bounds_file)

    log.info(f"Instantiating <{cfg.local_search._target_}>")
    local_search_module: LocalSearchModule = hydra.utils.instantiate(
        cfg.local_search, graph_module=graph_module, mutation_cfg=cfg.mutation
    )

    ind = local_search_module.initialize_individual()
    log.info(f"Initial individual {ind.fitness.values[0]}")

    hof = local_search_module.optimize(ind)
    if len(hof) > 0:
        best_ind = hof[0]
        best_ind_stats = hof.get_individual_stats(best_ind)

        log.info(f"Best fitness {best_ind.fitness.values[0]}")
        log.info(f"Best individual (iter, fitness, time_elapsed) {best_ind_stats}")
        log.info(f"Num clusters: {len(set(best_ind))}")

        assert best_ind.fitness.values == evaluate(best_ind)

        # add and remove edges from graph given the best individual
        graph_module.apply_edits(best_ind)
        log.info(f"Solution valid: {graph_module.is_valid_solution()}")

        if cfg.draw_graph:
            nx_final_graph = nx.Graph(graph_module.graph)
            nx.draw_networkx(nx_final_graph)
            plt.show()
    else:
        log.info("No feasible solution found")

    object_dict = {
        "cfg": cfg,
        "graph_module": graph_module,
        "local_search_module": local_search_module,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

        logger.experiment.run.summary["best_fitness"] = best_ind.fitness.values[0]
        logger.experiment.run.summary["iter"] = best_ind_stats[0]
        logger.experiment.run.summary["time_elapsed"] = best_ind_stats[2]
        logger.experiment.run.summary["num_clusters"] = len(set(best_ind))

    return local_search_module.logbook, object_dict


@hydra.main(
    version_base=None, config_path="../configs", config_name="local_search_experiment"
)
def main(cfg: DictConfig):
    print_config_tree(cfg)

    # get list of txt files in data folder
    input_files = sorted(os.listdir(cfg.data.data_dir))
    for input_file in input_files:
        log.info("Instantiating Weights and Biases...")
        logger: List[Logger] = instantiate_loggers(cfg.logger)[0]
        logger.experiment.name = input_file.replace(".txt", "")

        stats, object_dict = run(cfg, input_file=input_file)

        # Let's create an empty DataFrame
        df = pd.DataFrame()

        # Loop over each item in the data dictionary
        for key, values in stats.chapters.items():
            # Convert list of dictionaries to a DataFrame
            temp_df = pd.DataFrame(values)
            temp_df = temp_df.rename(columns={"min": f"{key}_min", "iter": "iter"})

            # If df is empty, copy temp_df to df
            if df.empty:
                df = temp_df
            # Otherwise, merge df and temp_df on the iter column
            else:
                df = pd.merge(df, temp_df, on="iter")

        # Set the index to be this and don't drop
        df.set_index("iter", inplace=True)
        df["fitness_min"] = df["fitness_min"].apply(itemgetter(0))

        stats_df = df

        # Save the DataFrame to a CSV file
        stats_dir = Path(cfg.paths.output_dir, "stats")
        stats_file = Path(
            stats_dir,
            input_file.replace("input", "stats").replace(".txt", ".csv"),
        )
        log.info(f"Saving stats to {Path(stats_dir, stats_file)}")
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)
        stats_df.to_csv(Path(stats_dir, stats_file))

        # For visualization purposes fitness needs to be normalized using the computed bounds
        bounds_dir = Path(cfg.paths.data_dir, "bounds")
        # load the corresponding bounds file for the input file
        bounds_file = Path(bounds_dir, "output" + input_file.replace("input", ""))
        with open(bounds_file) as f:
            l_bound, u_bound = [int(x) for x in f.readline().split()]

        stats_df["fitness_min"] = stats_df["fitness_min"].apply(
            lambda x: (x - l_bound) / (u_bound - l_bound)
        )

        # Visualize the stats with matplotlib plots
        if cfg.visualize:
            plot_dir = Path(cfg.paths.output_dir, "plots")
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            plot_file = Path(
                input_file.replace("input", "plot").replace(".txt", ".png")
            )
            save_plots(plot_file, stats_df, object_dict["local_search_module"].hof)


if __name__ == "__main__":
    main()
