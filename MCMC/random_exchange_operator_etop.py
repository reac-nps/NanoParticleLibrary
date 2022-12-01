from MCMC.RandomExchangeOperator import RandomExchangeOperator

import numpy as np
import itertools

class RandomExchangeOperatorExtended(RandomExchangeOperator):
    def __init__(self, p_geometric):
        self.symbols = None
        self.n_symbols_atoms = 0
        
        self.swap_types = 0
        self.n_swap_types = None
        self.swap_types_probability = None
        
        self.max_exchanges = 0
        self.p_geometric = p_geometric
        
    def bind_particle(self, particle):
        self.symbols = sorted(particle.atoms.get_all_symbols())
        
        symbols_indices = dict()
        for symbol in self.symbols:
            symbols_indices[symbol] = particle.get_indices_by_symbol(symbol)
        
        
        self.max_exchanges = min([len(x) for x in symbols_indices.values()])
        
        self.swap_types = [swap_type for swap_type in itertools.combinations(self.symbols, 2)]
        self.get_swap_porbability(symbols_indices)
        self.n_symbols_atoms = [len(symbols_indices[symbol]) for symbol in self.symbols]
        self.n_swap_types = len(self.swap_types)

    def get_swap_porbability(self, symbols_indices):
        self.swap_types_probability = np.zeros(len(self.swap_types))
        for i, swap_type in enumerate(self.swap_types):
            self.swap_types_probability[i] += len(symbols_indices[swap_type[0]])
            self.swap_types_probability[i] += len(symbols_indices[swap_type[1]])

        self.swap_types_probability = self.swap_types_probability/sum(self.swap_types_probability)        
    
    def random_exchange(self, particle):
        swap_type_probability = np.random.choice(self.n_swap_types, 1, p=self.swap_types_probability)[0]
        symbol1, symbol2 = self.swap_types[swap_type_probability]
        max_exchanges = 1
        n_exchanges = min(np.random.geometric(p=self.p_geometric, size=1)[0], max_exchanges)
        symbol1_indices = np.random.choice(particle.get_indices_by_symbol(symbol1), n_exchanges, replace=False)
        symbol2_indices = np.random.choice(particle.get_indices_by_symbol(symbol2), n_exchanges, replace=False)

        particle.swap_symbols(list(zip(symbol1_indices, symbol2_indices)))
        return list(zip(symbol1_indices, symbol2_indices))
