import random
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from operator import attrgetter

import rootutils
from omegaconf import DictConfig
from deap import base, creator, tools
import numpy as np
from tqdm.auto import tqdm
import networkx as nx
import time
from lightning.pytorch.loggers import WandbLogger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from operators.mutation import mutate_moveOrIsolateOrRemoveCliques
from operators.evaluation import evaluate
import src.utils.local_search as ls
from src.utils import pylogger
import src.utils.utils as utils
from utils.timed_halloffame import TimedHallOfFame
from src.utils.graph import compute_cluster_mapping
from src.modules.graph_module import GraphModule
from src.utils.utils import normalize

log = pylogger.get_pylogger(__name__)


class LocalSearchModule:
    def __init__(
        self,
        mutation_pb,
        num_iters,
        graph_module: GraphModule,
        verbose=False,
        mutation_cfg=None,
        logger: WandbLogger = None,
    ):
        self.graph_module = graph_module
        self.verbose = verbose
        self.mutation_pb = mutation_pb
        self.num_iters = num_iters
        self.mutation_cfg = mutation_cfg

        self.logger = logger

        graph_module.compute_clusters()

        # Create the fitness and individual classes
        if "FitnessMin" in creator.__dict__:
            del creator.FitnessMin
        if "Individual" in creator.__dict__:
            del creator.Individual

        creator.create(
            "FitnessMin", base.Fitness, weights=(-1.0,)
        )  # Minimization problem
        creator.create(
            "Individual",
            list,
            fitness=creator.FitnessMin,
            max_clusters=max(self.graph_module.components) + 1,
            num_nodes=len(graph_module.graph),
            cluster_mapping=defaultdict(set),
            graph=graph_module.graph,
        )
        self.toolbox = base.Toolbox()

        # Register the Evaluation Function and Constraints
        self.toolbox.register("evaluate", evaluate)
        if self.mutation_cfg is not None:
            self.toolbox.register(
                "mutate", mutate_moveOrIsolateOrRemoveCliques, **self.mutation_cfg
            )
        else:
            self.toolbox.register("mutate", mutate_moveOrIsolateOrRemoveCliques)

        # Initialize Statistics
        fit_stats = tools.Statistics(key=attrgetter("fitness.values"))
        num_clusters_stats = tools.Statistics(
            lambda ind: len(ind.cluster_mapping.keys())
        )

        self.stats = tools.MultiStatistics(
            fitness=fit_stats,
            num_clusters=num_clusters_stats,
        )
        self.stats.register("min", np.min, axis=0)
        self.stats.register(
            "norm_min",
            lambda pop: normalize(
                np.min(
                    pop,
                    axis=0,
                ),
                self.graph_module.l_bound,
                self.graph_module.u_bound,
            ),
        )

        # Initialize Hall of Fame
        self.hof = TimedHallOfFame(maxsize=10, similar=np.array_equal)

        # Initialize Logbook
        self.logbook = tools.Logbook()
        self.logbook.header = ("iter", "fitness", "num_clusters")
        self.logbook.chapters["fitness"].header = ("min", "norm_min")
        self.logbook.chapters["num_clusters"].header = ("min",)

    def log(self, iter, record: dict):
        if self.logger:
            self.logger.log_metrics(step=iter, metrics=record)

    def initialize_individual(self):
        ind = creator.Individual([i for i in range(len(self.graph_module.graph))])
        ind.cluster_mapping = compute_cluster_mapping(ind)
        ind.fitness.values = self.toolbox.evaluate(ind)
        return ind

    def optimize(self, ind):
        self.hof.start()
        self.hof.update([ind], 0)

        record = self.stats.compile([ind]) if self.stats else {}
        self.log(iter=0, record=record)
        self.logbook.record(iter=0, **record)
        if self.verbose:
            print(self.logbook.stream)
        try:
            for iter in range(1, self.num_iters):
                # perturb
                if random.random() < self.mutation_pb:
                    self.toolbox.mutate(ind)

                if not ind.fitness.valid:
                    print("Invalid fitness", ind.fitness.values[0])
                    ind.fitness.values = self.toolbox.evaluate(ind)

                # local search
                (ind,) = ls.label_propagation(ind, self.graph_module.graph)
                # print("2.", ind, ind.fitness.values[0])
                (ind,) = ls.clique_removal(ind, self.graph_module.graph)

                if self.hof is not None:
                    self.hof.update([ind], iter)

                # Append the current iteration statistics to the logbook
                record = self.stats.compile([ind]) if self.stats else {}
                self.log(iter=iter, record=record)
                self.logbook.record(iter=iter, **record)
                if self.verbose:
                    print(self.logbook.stream)

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected, gracefully stopping optimization.")

        return self.hof
