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
from lightning.pytorch.loggers import Logger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from operators.evaluation import evaluate
from src.utils import pylogger
import src.utils.utils as utils
from src.utils.instantiators import instantiate_loggers
from modules.graph_module import GraphModule
from modules.evolution_module import EvolutionModule
from src.utils.logging_utils import log_hyperparameters

from src.utils.utils import print_config_tree

log = pylogger.get_pylogger(__name__)


@utils.task_wrapper
def run(
    cfg: DictConfig,
    input_file: str = "0_cliques_ring.txt",
    bounds_file="0_cliques_ring.txt",
    logger=None,
):
    random.seed(cfg.seed)

    log.info(f"Instantiating <{cfg.data._target_}>")
    graph_module: GraphModule = hydra.utils.instantiate(cfg.data)

    # Load Graph data
    graph_module.load_graph(input_file, bounds_file)

    log.info(f"Instantiating <{cfg.evolution._target_}>")
    evolution_module: EvolutionModule = hydra.utils.instantiate(
        cfg.evolution,
        graph_module=graph_module,
        mutation_cfg=cfg.mutation,
        crossover_cfg=cfg.crossover,
        selection_cfg=cfg.selection,
        local_search_cfg=cfg.local_search,
        logger=logger,
    )

    log.info(f"Initializing population")
    evolution_module.initialize_population()

    log.info("Running evolution")
    hof = evolution_module.optimize()

    if len(hof) > 0:
        best_ind = hof[0]
        best_ind_stats = hof.get_individual_stats(best_ind)
        log.info(f"Best fitness {best_ind.fitness.values[0]}")
        log.info(f"Best individual (iter, fitness, time_elapsed) {best_ind_stats}")
        log.info(f"Num clusters: {len(set(best_ind))}")
        log.info(f"Fitness: {evaluate(best_ind)}")

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
        "evolution_module": evolution_module,
        "logger": logger,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

        logger.experiment.summary["best_fitness"] = best_ind.fitness.values[0]
        logger.experiment.summary["iter"] = best_ind_stats[0]
        logger.experiment.summary["time_elapsed"] = best_ind_stats[2]
        logger.experiment.summary["num_clusters"] = len(set(best_ind))

    return evolution_module.logbook, object_dict


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print_config_tree(cfg)

    # get list of txt files in data folder
    input_files = sorted(os.listdir(cfg.data.data_dir))
    for input_file in input_files:
        log.info("Instantiating Weights and Biases...")
        cfg.logger.wandb["group"] = "local_search"
        logger: List[Logger] = instantiate_loggers(cfg.logger)[0]
        logger.experiment.name = "ev_" + input_file.replace(".txt", "")

        if cfg.evolution.enable_local_search:
            logger.experiment.group = "evolution_ls"

        stats, object_dict = run(
            cfg, input_file=input_file, bounds_file=input_file, logger=logger
        )

        # Let's create an empty DataFrame
        df = pd.DataFrame()

        # Loop over each item in the data dictionary
        for key, values in stats.chapters.items():
            # Convert list of dictionaries to a DataFrame
            temp_df = pd.DataFrame(values)
            temp_df = temp_df.rename(
                columns={
                    "min": f"{key}_min",
                    "gen": "gen",
                    "avg": f"{key}_avg",
                    "std": f"{key}_std",
                    "nevals": "nevals",
                }
            )

            # If df is empty, copy temp_df to df
            if df.empty:
                df = temp_df
            # Otherwise, merge df and temp_df on the iter column
            else:
                df = pd.merge(df, temp_df, on=["gen", "nevals"])

        # Set the index to be this and don't drop
        df.set_index("gen", inplace=True)
        df["fitness_min"] = df["fitness_min"].apply(itemgetter(0))
        df["fitness_avg"] = df["fitness_avg"].apply(itemgetter(0))
        df["fitness_std"] = df["fitness_std"].apply(itemgetter(0))

        stats_df = df

        # Save the DataFrame to a CSV file
        stats_dir = Path(cfg.paths.output_dir, "stats")
        stats_file = Path(
            stats_dir,
            input_file.replace(".txt", ".csv"),
        )
        log.info(f"Saving stats to {Path(stats_dir, stats_file)}")
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)
        stats_df.to_csv(Path(stats_dir, stats_file))


if __name__ == "__main__":
    main()
