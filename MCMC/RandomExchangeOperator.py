import numpy as np


class RandomExchangeOperator:
    def __init__(self, p_geometric):
        self.symbol1 = None
        self.symbol2 = None

        self.n_symbol1_atoms = 0
        self.n_symbol2_atoms = 0

        self.n_adsorbates = 0
        self.n_adsorption_sites = 0

        self.max_exchanges = 0
        self.max_adsorbate_exchanges = 0
        self.p_geometric = p_geometric

    def bind_particle(self, particle):
        symbols = sorted(particle.atoms.get_all_symbols())
        self.symbol1 = symbols[0]
        if len(symbols) == 2:
            self.symbol2 = symbols[1]

        symbol1_indices = particle.get_indices_by_symbol(self.symbol1)
        symbol2_indices = particle.get_indices_by_symbol(self.symbol2)

        self.n_symbol1_atoms = len(symbol1_indices)
        self.n_symbol2_atoms = len(symbol2_indices)

        self.max_exchanges = min(self.n_symbol1_atoms, self.n_symbol2_atoms)

    def bind_adsorbates(self, particle, n_adsorbates):
        self.bind_particle(particle)
        particle.construct_adsorption_list()

        if particle.get_number_of_adsorbates() != n_adsorbates:
            particle.random_occupation(n_adsorbates)

        self.n_adsorbates = n_adsorbates
        self.n_adsorption_sites = particle.get_total_number_of_sites()
        self.max_adsorbate_exchanges = n_adsorbates

    def random_exchange(self, particle):
        n_exchanges = min(np.random.geometric(p=self.p_geometric, size=1)[0], self.max_exchanges)
        symbol1_indices = np.random.choice(particle.get_indices_by_symbol(self.symbol1), n_exchanges, replace=False)
        symbol2_indices = np.random.choice(particle.get_indices_by_symbol(self.symbol2), n_exchanges, replace=False)

        particle.swap_symbols(zip(symbol1_indices, symbol2_indices))
        return list(zip(symbol1_indices, symbol2_indices))

    def random_adsorbate_migration(self, particle):

        n_exchanges = min(np.random.geometric(p=self.p_geometric, size=1)[0], self.max_adsorbate_exchanges)
        status0_indices = np.random.choice(particle.get_occupation_status_by_indices(0), n_exchanges, replace=False)
        status1_indices = np.random.choice(particle.get_occupation_status_by_indices(1), n_exchanges, replace=False)

        particle.swap_status(zip(status0_indices, status1_indices))
        return list(zip(status0_indices, status1_indices))

    def coupled_random_exchange(self, particle):
        probability = np.random.randint(2)
        n_exchanges = min(np.random.geometric(p=self.p_geometric, size=1)[0], self.max_exchanges)
        symbol1_indices = np.random.choice(particle.get_indices_by_symbol(self.symbol1), n_exchanges*probability, replace=False)
        symbol2_indices = np.random.choice(particle.get_indices_by_symbol(self.symbol2), n_exchanges*probability, replace=False)

        ordering_exchange = list(zip(symbol1_indices, symbol2_indices))
        particle.swap_symbols(ordering_exchange)

        n_exchanges = min(np.random.geometric(p=self.p_geometric, size=1)[0], self.max_adsorbate_exchanges)
        status0_indices = np.random.choice(particle.get_occupation_status_by_indices(0), n_exchanges*(1-probability), replace=False)
        status1_indices = np.random.choice(particle.get_occupation_status_by_indices(1), n_exchanges*(1-probability), replace=False)

        adsorbate_exchange = list(zip(status0_indices, status1_indices))
        particle.swap_status(adsorbate_exchange)

        return ordering_exchange, adsorbate_exchange
