from sortedcontainers import SortedKeyList


# TODO rename, make sure variable names align with thesis
class GuidedExchangeOperator:
    def __init__(self, environment_energies, feature_key):
        self.n_envs = int(len(environment_energies)/2)
        self.env_energy_differences = [environment_energies[i] - environment_energies[i + self.n_envs] for i in range(self.n_envs)]

        self.feature_key = feature_key

        self.symbol1_exchange_energies = dict()
        self.symbol2_exchange_energies = dict()

        self.symbol1_indices = SortedKeyList(key=lambda x: self.symbol1_exchange_energies[x])
        self.symbol2_indices = SortedKeyList(key=lambda x: self.symbol2_exchange_energies[x])

        self.n_symbol1_atoms = 0
        self.n_symbol2_atoms = 0

    def env_from_feature(self, x):
        return x % self.n_envs

    def guided_exchange(self, particle):
        symbol1_index = self.symbol1_indices[0]
        symbol2_index = self.symbol2_indices[0]

        particle.swap_symbols([(symbol1_index, symbol2_index)])
        return symbol1_index, symbol2_index

    def basin_hop_step(self, particle):
        expected_energy_gain = -1
        index = 0
        while expected_energy_gain <= 0 and index < min(self.n_symbol1_atoms, self.n_symbol2_atoms):
            index += 1
            symbol1_index = self.symbol1_indices[index % self.n_symbol1_atoms]
            symbol1_energy = self.symbol1_exchange_energies[symbol1_index]

            symbol2_index = self.symbol2_indices[index % self.n_symbol2_atoms]
            symbol2_energy = self.symbol2_exchange_energies[symbol2_index]

            expected_energy_gain = symbol1_energy + symbol2_energy
            if expected_energy_gain > 0:
                particle.swap_symbols([(symbol1_index, symbol2_index)])
                return symbol1_index, symbol2_index

        symbol1_index = self.symbol1_indices[index % self.n_symbol1_atoms]
        symbol2_index = self.symbol2_indices[index % self.n_symbol2_atoms]

        particle.swap_symbols([(symbol1_index, symbol2_index)])
        return symbol1_index, symbol2_index

    def bind_particle(self, particle):
        symbols = sorted(particle.atoms.get_all_symbols())
        symbol1 = symbols[0]
        symbol2 = symbols[1]

        symbol1_indices = particle.get_indices_by_symbol(symbol1)
        symbol2_indices = particle.get_indices_by_symbol(symbol2)

        self.n_symbol1_atoms = len(symbol1_indices)
        self.n_symbol2_atoms = len(symbol2_indices)

        atom_features = particle.get_atom_features(self.feature_key)
        for index in symbol1_indices:
            feature = atom_features[index]
            self.symbol1_exchange_energies[index] = -self.env_energy_differences[self.env_from_feature(feature)]
            self.symbol1_indices.add(index)

        for index in symbol2_indices:
            feature = atom_features[index]
            self.symbol2_exchange_energies[index] = +self.env_energy_differences[self.env_from_feature(feature)]
            self.symbol2_indices.add(index)

    def update(self, particle, indices, exchange_indices):
        symbols = sorted(particle.atoms.get_all_symbols())
        symbol1 = symbols[0]

        atom_features = particle.get_atom_features(self.feature_key)
        for index in indices:
            if index in exchange_indices:
                if particle.get_symbol(index) == symbol1:
                    self.symbol2_indices.remove(index)
                else:
                    self.symbol1_indices.remove(index)
            else:
                if particle.get_symbol(index) == symbol1:
                    self.symbol1_indices.remove(index)
                else:
                    self.symbol2_indices.remove(index)

        for index in indices:
            feature = atom_features[index]
            new_exchange_energy = self.env_energy_differences[self.env_from_feature(feature)]
            if index in exchange_indices:
                if particle.get_symbol(index) == symbol1:
                    self.symbol1_exchange_energies[index] = -new_exchange_energy
                    del self.symbol2_exchange_energies[index]
                else:
                    self.symbol2_exchange_energies[index] = new_exchange_energy
                    del self.symbol1_exchange_energies[index]
            else:
                if particle.get_symbol(index) == symbol1:
                    self.symbol1_exchange_energies[index] = -new_exchange_energy
                else:
                    self.symbol2_exchange_energies[index] = new_exchange_energy

        for index in indices:
            if particle.get_symbol(index) == symbol1:
                self.symbol1_indices.add(index)
            else:
                self.symbol2_indices.add(index)
