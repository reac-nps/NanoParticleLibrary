from sortedcontainers import SortedKeyList
import random 

class GuidedExchangeOperator:
    def __init__(self, environment_energies, feature_key):
        self.n_envs = int(len(environment_energies)/2)
        self.env_energy_differences = [environment_energies[i] - environment_energies[i + self.n_envs] for i in range(self.n_envs)]

        self.neigh_energy_difference = [environment_energies[i+1] - environment_energies[i] for i in range((self.n_envs*2)-1)] + [-1000]

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
        index = -1
        random_indices = random.sample(range(max(self.n_symbol1_atoms, self.n_symbol2_atoms)), max(self.n_symbol1_atoms, self.n_symbol2_atoms))
        
        while expected_energy_gain <= 0 and index < min(self.n_symbol1_atoms, self.n_symbol2_atoms):
            index += 1
            symbol1_index = self.symbol1_indices[random_indices[index] % self.n_symbol1_atoms]
            symbol1_energy = self.symbol1_exchange_energies[symbol1_index]

            symbol2_index = self.symbol2_indices[random_indices[index] % self.n_symbol2_atoms]
            symbol2_energy = self.symbol2_exchange_energies[symbol2_index]

            expected_energy_gain = symbol1_energy + symbol2_energy
            if expected_energy_gain > 0:
                particle.swap_symbols([(symbol1_index, symbol2_index)])
                return symbol1_index, symbol2_index

        symbol1_index = self.symbol1_indices[random_indices[index] % self.n_symbol1_atoms]
        symbol2_index = self.symbol2_indices[random_indices[index] % self.n_symbol2_atoms]

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
            
            energy_change = 0
            for neigh_idx in particle.get_coordination_atoms(index):
                neigh_feature = atom_features[neigh_idx]
                energy_change -= self.neigh_energy_difference[neigh_feature-1] 

            self.symbol1_exchange_energies[index] = (-self.env_energy_differences[self.env_from_feature(feature)] + energy_change)
            self.symbol1_indices.add(index)

        for index in symbol2_indices:
            feature = atom_features[index]
            
            energy_change = 0
            for neigh_idx in particle.get_coordination_atoms(index):
                neigh_feature = atom_features[neigh_idx]
                energy_change += self.neigh_energy_difference[neigh_feature] 

            self.symbol2_exchange_energies[index] = (+self.env_energy_differences[self.env_from_feature(feature)] + energy_change)
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
                    
                    energy_change = 0
                    for neigh_idx in particle.get_coordination_atoms(index):
                        neigh_feature = atom_features[neigh_idx]
                        energy_change -= self.neigh_energy_difference[neigh_feature-1]

                    self.symbol1_exchange_energies[index] = -new_exchange_energy + energy_change
                    del self.symbol2_exchange_energies[index]
                else:
                    energy_change = 0
                    for neigh_idx in particle.get_coordination_atoms(index):
                        neigh_feature = atom_features[neigh_idx]
                        energy_change += self.neigh_energy_difference[neigh_feature] 

                    self.symbol2_exchange_energies[index] = new_exchange_energy + energy_change
                    del self.symbol1_exchange_energies[index]
            else:
                if particle.get_symbol(index) == symbol1:

                    energy_change = 0
                    for neigh_idx in particle.get_coordination_atoms(index):
                        neigh_feature = atom_features[neigh_idx]
                        energy_change -= self.neigh_energy_difference[neigh_feature-1]

                    self.symbol1_exchange_energies[index] = -new_exchange_energy + energy_change
                else:
                    energy_change = 0
                    for neigh_idx in particle.get_coordination_atoms(index):
                        neigh_feature = atom_features[neigh_idx]
                        energy_change += self.neigh_energy_difference[neigh_feature] 
                    self.symbol2_exchange_energies[index] = new_exchange_energy + energy_change

        for index in indices:
            if particle.get_symbol(index) == symbol1:
                self.symbol1_indices.add(index)
            else:
                self.symbol2_indices.add(index)

    # def update(self, particle, indices, exchange_indices):
    #     symbols = sorted(particle.atoms.get_all_symbols())
    #     symbol1 = symbols[0]

    #     atom_features = particle.get_atom_features(self.feature_key)
    #     for index in indices:
    #         if index in exchange_indices:
    #             if particle.get_symbol(index) == symbol1:
    #                 self.symbol2_indices.remove(index)
    #                 #del self.symbol2_exchange_energies[index]
    #             else:
    #                 self.symbol1_indices.remove(index)
    #                 #del self.symbol1_exchange_energies[index]
    #         else:
    #             if particle.get_symbol(index) == symbol1:
    #                 self.symbol1_indices.remove(index)
    #             else:
    #                 self.symbol2_indices.remove(index)

    #     for index in indices:
    #         feature = atom_features[index]
    #         new_exchange_energy = self.env_energy_differences[self.env_from_feature(feature)]
    #         if particle.get_symbol(index) == symbol1:
    #             energy_change = 0
    #             for neigh_idx in particle.get_coordination_atoms(index):
    #                 neigh_feature = atom_features[neigh_idx]
    #                 energy_change -= self.neigh_energy_difference[neigh_feature-1]
    #             #self.symbol1_exchange_energies[index] = (-self.env_energy_differences[self.env_from_feature(feature)] + energy_change)
    #             #elf.symbol1_indices.add(index)
    #         else:
    #             energy_change = 0
    #             for neigh_idx in particle.get_coordination_atoms(index):
    #                 neigh_feature = atom_features[neigh_idx]
    #                 energy_change += self.neigh_energy_difference[neigh_feature+1]
    #             #self.symbol2_exchange_energies[index] = (+self.env_energy_differences[self.env_from_feature(feature)] + energy_change)
    #             #self.symbol2_indices.add(index)
            
    #         if index in exchange_indices:
    #             if particle.get_symbol(index) == symbol1:
    #                 self.symbol1_exchange_energies[index] = -new_exchange_energy + energy_change
    #                 del self.symbol2_exchange_energies[index]
    #             else:
    #                 self.symbol2_exchange_energies[index] = new_exchange_energy + energy_change
    #                 del self.symbol1_exchange_energies[index]
    #         else:
    #             if particle.get_symbol(index) == symbol1:
    #                 self.symbol1_exchange_energies[index] = -new_exchange_energy + energy_change
    #             else:
    #                 self.symbol2_exchange_energies[index] = new_exchange_energy + energy_change

    #     for index in indices:
    #         if particle.get_symbol(index) == symbol1:
    #             self.symbol1_indices.add(index)
    #         else:
    #             self.symbol2_indices.add(index)