import numpy as np
from sklearn.cluster import KMeans
import copy


class LocalEnvironmentFeatureClassifier:
    def __init__(self, local_environment_calculator):
        self.local_environment_calculator = local_environment_calculator
        self.feature_key = None

    def compute_atom_features(self, particle, recompute_local_environments=False):
        if recompute_local_environments:
            self.local_environment_calculator.compute_local_environments(particle)

        for atom_index in particle.get_indices():
            self.compute_atom_feature(particle, atom_index, recompute_local_environments)

    def compute_feature_vector(self, particle, recompute_atom_features=True, recompute_local_environments=False):
        if recompute_atom_features:
            self.compute_atom_features(particle, recompute_local_environments)

        n_features = self.compute_n_features(particle)
        feature_vector = np.zeros(n_features)
        atom_features = particle.get_atom_features(self.feature_key)
        for index in particle.get_indices():
            feature_vector[atom_features[index]] += 1

        particle.set_feature_vector(self.feature_key, feature_vector)

    def compute_atom_feature(self, particle, atom_index, recompute_local_environment=False):
        feature = self.predict_atom_feature(particle, atom_index, recompute_local_environment)
        atom_features = particle.get_atom_features(self.feature_key)
        atom_features[atom_index] = feature

    def get_feature_key(self):
        return self.feature_key

    def set_feature_key(self, feature_key):
        self.feature_key = feature_key

    def compute_n_features(self, particle):
        raise NotImplementedError

    def predict_atom_feature(self, particle, lattice_index, recompute_local_environment=False):
        raise NotImplementedError


class KMeansClassifier(LocalEnvironmentFeatureClassifier):
    def __init__(self, n_cluster, local_environment_calculator, feature_key):
        LocalEnvironmentFeatureClassifier.__init__(self, local_environment_calculator)
        self.kMeans = None
        self.n_cluster = n_cluster
        self.feature_key = feature_key

    def compute_n_features(self, particle):
        n_elements = len(particle.get_all_symbols())
        n_features = self.n_cluster * n_elements
        return n_features

    # TODO problematic: length of feature vector different for pure particles and bimetallic ones
    def predict_atom_feature(self, particle, atom_index, recompute_local_environment=False):
        symbol = particle.get_symbol(atom_index)
        symbols = sorted(particle.get_all_symbols())
        symbol_index = symbols.index(symbol)

        offset = symbol_index*self.n_cluster
        if recompute_local_environment:
            environment = self.kMeans.predict([self.local_environment_calculator.predict_local_environment(particle, atom_index)])[0]
        else:
            environment = self.kMeans.predict([particle.get_local_environment(atom_index)])[0]
        return offset + environment

    def kmeans_clustering(self, training_set):
        local_environments = list()
        for particle in training_set:
            local_environments = local_environments + list(particle.get_local_environments().values())

        print("Starting kMeans")
        self.kMeans = KMeans(n_clusters=self.n_cluster, random_state=0).fit(local_environments)


# TODO rename, rework with offsets etc.
class TopologicalEnvironmentClassifier(LocalEnvironmentFeatureClassifier):
    def __init__(self, local_environment_calculator, symbols):
        LocalEnvironmentFeatureClassifier.__init__(self, local_environment_calculator)
        symbols_copy = copy.deepcopy(symbols)
        symbols_copy.sort()
        self.symbols = symbols_copy

        self.coordination_number_offsets = [int(cn*(cn + 1)/2) for cn in range(13)]

        self.feature_key = 'TEC'

    def compute_n_features(self, particle):
        return 182

    def predict_atom_feature(self, particle, atom_index, recompute_local_environment=False):
        symbol = particle.get_symbol(atom_index)
        symbol_index = self.symbols.index(symbol)

        element_offset = symbol_index*91

        if recompute_local_environment:
            self.local_environment_calculator.compute_local_environment(particle, atom_index)

        environment = particle.get_local_environment(atom_index)
        coordination_number = len(particle.neighbor_list[atom_index])

        atom_feature = int(element_offset + self.coordination_number_offsets[coordination_number] + environment[0])

        return atom_feature


# TODO rename
class CoordinationNumberClassifier(LocalEnvironmentFeatureClassifier):
    def __init__(self, local_environment_calculator):
        LocalEnvironmentFeatureClassifier.__init__(self, local_environment_calculator)
        self.coordination_number_offsets = [int(cn * (cn + 1) / 2) for cn in range(13)]

        self.feature_key = 'TEC'

    def compute_n_features(self, particle):
        return 182

    def predict_atom_feature(self, particle, atom_index, recompute_local_environment=False):
        symbol = particle.get_symbol(atom_index)
        if symbol == 'X':
            symbol_index = 1
        else:
            symbol_index = 0

        element_offset = symbol_index*91

        if recompute_local_environment:
            self.local_environment_calculator.compute_local_environment(particle, atom_index)

        environment = particle.get_local_environment(atom_index)
        coordination_number = environment[0]  # TODO not robust, only works if 'X' as 'empty site' is second entry
        # TODO should specify index of non-vacancy element

        atom_feature = element_offset + self.coordination_number_offsets[coordination_number] + environment[0]

        return atom_feature
