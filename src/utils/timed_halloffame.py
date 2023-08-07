import time
from deap import tools


class TimedHallOfFame(tools.HallOfFame):
    def __init__(self, *args, **kwargs):
        super(TimedHallOfFame, self).__init__(*args, **kwargs)
        self.history = {}
        self.start_time = None
        self.gen = 0

    def start(self):
        self.start_time = time.time()

    def insert(self, item):
        super(TimedHallOfFame, self).insert(item)
        # Convert the individual to string for use as dictionary key
        ind_repr = str(item)
        timestamp = time.time() - self.start_time
        if ind_repr not in self.history:
            self.history[ind_repr] = (self.gen, item.fitness.values[0], timestamp)

    def update(self, population, gen):
        self.gen = gen
        super(TimedHallOfFame, self).update(population)

    def get_individual_stats(self, ind):
        ind_repr = str(ind)
        return self.history.get(ind_repr, None)
