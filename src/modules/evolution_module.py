from deap import base, creator, tools, algorithms
import numpy as np
import rootutils
from collections import defaultdict
import random
from operator import attrgetter
from tqdm.auto import tqdm
from lightning.pytorch.loggers import WandbLogger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import pylogger
from operators.evaluation import evaluate
from utils.timed_halloffame import TimedHallOfFame
import src.utils.utils as utils
import src.utils.local_search as ls
from src.utils.graph import compute_cluster_mapping
from src.operators.strategies import varAnd
from src.operators.mutation import mutate_moveOrIsolateOrRemoveCliques
from src.operators.crossover import cxCommonCluster
from src.modules.graph_module import GraphModule
from src.utils.utils import normalize

log = pylogger.get_pylogger(__name__)


class EvolutionModule:
    def __init__(
        self,
        num_gens: int,
        pop_size: int,
        mu: int,
        use_seed_individual: str,
        max_clusters: int,
        enable_local_search: bool,
        graph_module: GraphModule,
        mutation_cfg,
        crossover_cfg,
        selection_cfg,
        local_search_cfg,
        verbose: bool = False,
        logger: WandbLogger = None,
    ):
        self.num_gens = num_gens
        self.pop_size = pop_size
        self.mu = mu
        self.max_clusters = max(int(max_clusters * len(graph_module.graph)), 1)
        self.verbose = verbose
        self.use_seed_individual = use_seed_individual
        self.enable_local_search = enable_local_search

        self.logger = logger

        self.mutation_cfg = mutation_cfg
        self.crossover_cfg = crossover_cfg
        self.selection_cfg = selection_cfg

        self.mutation_pb = mutation_cfg.mutation_pb
        self.crossover_pb = crossover_cfg.crossover_pb

        if self.enable_local_search:
            self.local_search_cfg = local_search_cfg
            self.local_search_pb = local_search_cfg.local_search_pb

        self.graph_module: GraphModule = graph_module
        graph_module.compute_clusters()

        # Create the Fitness and Individual classes
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

        # Define the Decision Variables
        self.toolbox.register(
            "attr_int",
            random.randint,
            0,
            self.max_clusters - 1,
        )  # Integer values [0, max_clusters]
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_int,
            n=len(graph_module.graph),
        )

        # Define the Population
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # Register the Evaluation Function and Constraints
        self.toolbox.register("evaluate", evaluate)

        self.toolbox.register(
            "mate",
            cxCommonCluster,
            toolbox=self.toolbox,
            **self.crossover_cfg.cxCommonCluster,
        )
        self.toolbox.register(
            "mutate",
            mutate_moveOrIsolateOrRemoveCliques,
            **self.mutation_cfg.mutate_moveOrIsolateOrRemoveCliques,
        )

        self.toolbox.register("select", tools.selTournament, **self.selection_cfg)

        # Set up Statistics and Hall of Fame
        self.hof = TimedHallOfFame(maxsize=10, similar=np.array_equal)
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

        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
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
        self.stats.register(
            "norm_avg",
            lambda pop: normalize(
                np.mean(
                    pop,
                    axis=0,
                ),
                self.graph_module.l_bound,
                self.graph_module.u_bound,
            ),
        )

        # Initialize Logbook
        self.logbook = tools.Logbook()
        self.logbook.header = ("gen", "fitness", "num_clusters")
        self.logbook.chapters["fitness"].header = (
            "min",
            "avg",
            "norm_min",
            "norm_avg",
        )
        self.logbook.chapters["num_clusters"].header = (
            "min",
            "avg",
        )

    def log(self, gen, record: dict):
        if self.logger:
            self.logger.log_metrics(step=gen, metrics=record)

    def initialize_population(self):
        self.population = self.toolbox.population(n=self.pop_size)

        if self.use_seed_individual == "cc":
            self.population[0] = creator.Individual(self.graph_module.components)
        elif self.use_seed_individual == "one_cluster":
            self.population[0] = creator.Individual([0] * len(self.graph_module.graph))
        elif self.use_seed_individual == "all":
            self.population[0] = creator.Individual(
                [i for i in range(len(self.graph_module.graph))]
            )
            self.population[1] = creator.Individual(self.graph_module.components)
            self.population[2] = creator.Individual([0] * len(self.graph_module.graph))

    def optimize(self):
        self.hof.start()

        for ind in self.population:
            ind.cluster_mapping = compute_cluster_mapping(ind)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if self.hof is not None:
            self.hof.update(self.population, gen=0)

        record = self.stats.compile(self.population) if self.stats else {}
        self.log(gen=0, record=record)
        self.logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if self.verbose:
            print(self.logbook.stream)

        # Run the Evolutionary Algorithm
        try:
            for gen in range(1, self.num_gens):
                self.population = self.toolbox.select(
                    self.population, k=len(self.population)
                )

                offspring = varAnd(
                    self.population,
                    self.toolbox,
                    self.pop_size,
                    self.crossover_pb,
                    self.mutation_pb,
                )

                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fits = self.toolbox.map(self.toolbox.evaluate, invalid_ind)

                for fit, ind in zip(fits, invalid_ind):
                    ind.fitness.values = fit

                # local search (very computationallty expensive)
                if self.enable_local_search:
                    for ind in tqdm(offspring, "Local search", leave=False):
                        if random.random() < self.local_search_pb:
                            ls.label_propagation(
                                ind,
                                self.graph_module.graph,
                                self.local_search_cfg.label_propagation_subset,
                            )
                            ls.clique_removal(
                                ind,
                                self.graph_module.graph,
                                self.local_search_cfg.clique_removal_subset,
                            )

                if self.hof is not None:
                    self.hof.update(offspring, gen)

                # Select the next generation population
                self.population[:] = self.toolbox.select(
                    self.population + offspring, self.mu
                )

                # Append the current generation statistics to the logbook
                record = self.stats.compile(self.population) if self.stats else {}
                self.log(gen=gen, record=record)
                self.logbook.record(gen=gen, nevals=len(invalid_ind), **record)
                if self.verbose:
                    print(self.logbook.stream)
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected, gracefully stopping optimization.")

        return self.hof
