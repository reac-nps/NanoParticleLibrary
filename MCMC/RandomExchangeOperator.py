import numpy as np


class RandomExchangeOperator:
    def __init__(self, p_geometric):
        self.symbol1 = None
        self.symbol2 = None

        self.n_symbol1_atoms = 0
        self.n_symbol2_atoms = 0

        self.max_exchanges = 0
        self.p_geometric = p_geometric

    def bind_particle(self, particle):
        symbols = sorted(particle.atoms.get_all_symbols())
        self.symbol1 = symbols[0]
        self.symbol2 = symbols[1]

        symbol1_indices = particle.get_indices_by_symbol(self.symbol1)
        symbol2_indices = particle.get_indices_by_symbol(self.symbol2)

        self.n_symbol1_atoms = len(symbol1_indices)
        self.n_symbol2_atoms = len(symbol2_indices)

        self.max_exchanges = min(self.n_symbol1_atoms, self.n_symbol2_atoms)

    def random_exchange(self, particle):
        n_exchanges = min(np.random.geometric(p=self.p_geometric, size=1)[0], self.max_exchanges)
        symbol1_indices = np.random.choice(particle.get_indices_by_symbol(self.symbol1), n_exchanges, replace=False)
        symbol2_indices = np.random.choice(particle.get_indices_by_symbol(self.symbol2), n_exchanges, replace=False)

        particle.swap_symbols(zip(symbol1_indices, symbol2_indices))
        return list(zip(symbol1_indices, symbol2_indices))

        
    def random_exchange_plus_adsorbate_migration(self, particle, adsorbate):
        n_exchanges = min(np.random.geometric(p=self.p_geometric, size=1)[0], self.max_exchanges)
        symbol1_indices = np.random.choice(particle.get_indices_by_symbol(self.symbol1), n_exchanges, replace=False)
        symbol2_indices = np.random.choice(particle.get_indices_by_symbol(self.symbol2), n_exchanges, replace=False)

        adsorbate_positions = np.random.choice(np.arange(len(adsorbate['site_list'])), adsorbate['number_of_ads'], replace=False)
        particle.swap_symbols(zip(symbol1_indices, symbol2_indices))

        return list(zip(symbol1_indices, symbol2_indices)), adsorbate_positions