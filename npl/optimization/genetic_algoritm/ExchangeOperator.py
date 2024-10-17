import numpy as np
import copy


class ExchangeOperator:
    def __init__(self, p):
        self.p = p
        self.index = 0

    def random_exchange(self, particle, n_exchanges=None):
        new_particle = copy.deepcopy(particle)
        if new_particle.is_pure():
            print("Pure particle! No permutation possible")
            return new_particle

        symbols = new_particle.get_all_symbols()
        symbol1 = symbols[0]
        symbol2 = symbols[1]

        max_exchanges = min(new_particle.get_n_atoms_of_symbol(symbol1), new_particle.get_n_atoms_of_symbol(symbol2))
        if n_exchanges is None:
            n_exchanges = min(np.random.geometric(p=self.p, size=1)[0], max_exchanges)

        symbol1_indices = np.random.choice(new_particle.get_indices_by_symbol(symbol1), n_exchanges, replace=False)
        symbol2_indices = np.random.choice(new_particle.get_indices_by_symbol(symbol2), n_exchanges, replace=False)

        new_particle.swap_symbols(zip(symbol1_indices, symbol2_indices))

        return new_particle
