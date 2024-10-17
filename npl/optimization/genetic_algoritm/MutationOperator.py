import numpy as np
import copy


class MutationOperator:
    def __init__(self, p, symbols):
        self.p = p
        self.symbols = set(symbols)

    def random_mutation(self, particle, n_mutations=None, symbols=None):
        new_particle = copy.deepcopy(particle)
        if new_particle.is_pure():
            symbol_from = new_particle.atoms.get_all_symbols()[0]
            print(symbol_from)
            other_symbols = list(self.symbols.difference({symbol_from}))
            print(other_symbols)
            symbol_to = np.random.choice(other_symbols, 1)[0]
            print(symbol_to)
        else:
            if symbols is None:
                symbols = np.random.choice(new_particle.atoms.get_all_symbols(), 2, replace=False)
                symbol_from = symbols[0]
                symbol_to = symbols[1]
            else:
                symbol_from = symbols[0]
                symbol_to = symbols[1]

        max_mutations = min(new_particle.get_stoichiometry().values())
        if n_mutations is None:
            n_mutations = min(self.draw_from_geometric_distribution(), max_mutations)

        atom_indices_to_be_transformed = np.random.choice(new_particle.atoms.get_indices_by_symbol(symbol_from), n_mutations, replace=False)

        new_particle.atoms.transform_atoms(zip(atom_indices_to_be_transformed, [symbol_to] * n_mutations), a)

        return new_particle

    def draw_from_geometric_distribution(self):
        return np.random.geometric(p=self.p, size=1)[0]
