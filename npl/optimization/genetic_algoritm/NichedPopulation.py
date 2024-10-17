import numpy as np


# TODO these features are experimental right now
class Population:
    def __init__(self):
        pass

    def add_offspring(self, particle):
        raise NotImplementedError

    def compute_fitness(self, particle):
        raise NotImplementedError

    def random_selection(self, n_individuals):
        raise NotImplementedError

    def tournament_selection(self, n_individuals, tournament_size):
        raise NotImplementedError

    def gaussian_tournament(self, n_individuals, tournament_size, mean=None):
        raise NotImplementedError


class NichedPopulation(Population):
    def __init__(self, niche_symbol, n_atoms):
        Population.__init__(self)

        self.niche_symbol = niche_symbol
        self.n_niches = n_atoms + 1
        self.population = dict()

    def __getitem__(self, item):
        return self.population[item]

    def compute_fitness(self, particle):
        niche = self.get_niche(particle)

        #return particle.get_energy('BRR')
        return 1

    def get_niche(self, particle):
        niche = particle.get_n_atoms_of_symbol(self.niche_symbol)

        return niche

    def add_offspring(self, particle):
        niche = self.get_niche(particle)

        self.population[niche] = particle

    def random_selection(self, n_individuals):
        return np.random.choice(list(self.population.values()), n_individuals, replace=False).tolist()

    def tournament_selection(self, n_individuals, tournament_size):
        winners = list()
        for tournament in range(n_individuals):
            candidates = np.random.choice(list(set(self.population.values()).difference(set(winners))), tournament_size, replace=False).tolist()
            candidates.sort(key=lambda x: self.compute_fitness(x))
            winners.append(candidates[0])

        return winners

    def gaussian_tournament(self, n_individuals, tournament_size, mean=None):
        winners = list()
        for tournament in range(n_individuals):
            if mean is None:
                mean = int(np.random.random() * self.n_niches)

            sigma = 5

            probabilities = np.array([1 / (sigma * 2 * np.pi) * np.exp(-0.5 * ((i - mean) / sigma) ** 2) for i in range(0, self.n_niches)])
            probabilities = probabilities / np.sum(probabilities)

            candidates = np.random.choice(list(self.population.values()), tournament_size, replace=False, p=probabilities).tolist()
            candidates.sort(key=lambda x: self.compute_fitness(x))

            winners.append(candidates[0])

        return winners

    def get_as_list(self):
        return list(self.population.values())

    def get_convex_hull(self):
        return [(niche, self.population[niche].get_energy('BRR')) for niche in range(self.n_niches)]


