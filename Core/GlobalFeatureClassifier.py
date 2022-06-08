from collections import defaultdict
import numpy as np
import itertools
import copy


class GlobalFeatureClassifier:
    """Base class for classifiers that produce a feature vector directly from a particle without the need
    for precomputing local environments.

    Valid implementations have to implement the compute_feature_vector(particle) method.
    """
    def __init__(self):
        self.feature_key = None

    def get_feature_key(self):
        return self.feature_key

    def set_feature_key(self, feature_key):
        self.feature_key = feature_key

    def compute_feature_vector(self, particle):
        raise NotImplementedError


# TODO better name
# TODO refactor handling of symbols
class SimpleFeatureClassifier(GlobalFeatureClassifier):
    """Base class for classifiers which provides handling of two elements and calculation of bond counts.

    Entries for different elements in feature vectors have to be ordered consistently. This class uses an
     alphabetical ordering of TWO elements. The returned feature vector consists of [n_aa_bonds/n_atoms,
     n_ab_bonds/n_atoms, n_bb_bonds/n_atoms, n_a_atoms*0.1].
    """
    def __init__(self, symbols):
        GlobalFeatureClassifier.__init__(self)
        symbols_copy = copy.deepcopy(symbols)
        symbols_copy.sort()
        self.symbol_a = symbols_copy[0]
        if len(symbols) == 2:
            self.symbol_b = symbols_copy[1]

        self.feature_key = 'SFC'
        return

    def compute_feature_vector(self, particle):
        n_aa_bonds, n_bb_bonds, n_ab_bonds = self.compute_respective_bond_counts(particle)
        n_atoms = particle.atoms.get_n_atoms()

        M = particle.get_stoichiometry()[self.symbol_a] * 0.1
        particle.set_feature_vector(self.feature_key, np.array([n_aa_bonds / n_atoms, n_bb_bonds / n_atoms, n_ab_bonds / n_atoms, M]))

    def compute_respective_bond_counts(self, particle):
        n_aa_bonds = 0
        n_ab_bonds = 0
        n_bb_bonds = 0

        for lattice_index_with_symbol_a in particle.atoms.get_indices_by_symbol(self.symbol_a):
            neighbor_list = particle.neighbor_list[lattice_index_with_symbol_a]
            for neighbor in neighbor_list:
                symbol_neighbor = particle.atoms.get_symbol(neighbor)

                if self.symbol_a != symbol_neighbor:
                    n_ab_bonds += 0.5
                else:
                    n_aa_bonds += 0.5

        for lattice_index_with_symbol_b in particle.atoms.get_indices_by_symbol(self.symbol_b):
            neighbor_list = particle.neighbor_list[lattice_index_with_symbol_b]
            for neighbor in neighbor_list:
                symbol_neighbor = particle.atoms.get_symbol(neighbor)

                if self.symbol_b == symbol_neighbor:
                    n_bb_bonds += 0.5
                else:
                    n_ab_bonds += 0.5

        return n_aa_bonds, n_bb_bonds, n_ab_bonds

class testTopologicalFeatureClassifier(SimpleFeatureClassifier):
    """Classifier for a generalization of the topological descriptors by Kozlov et al. (2015).
    Implemented for TWO elements, which are sorted alphabetically.The returned feature vector will have the form
    [n_aa_bonds/n_atoms, n_ab_bonds/n_atoms, n_bb_bonds/n_atoms, n_a_atoms*0.1, n_a(cn=0), n_a(cn=1), ..., n_a(cn=12].
    """
    def __init__(self, symbols):
        SimpleFeatureClassifier.__init__(self, symbols)
        self.feature_key = 'TFC'

    def compute_feature_vector(self, particle):
        n_atoms = particle.get_n_atoms()
        n_aa_bonds, n_bb_bonds, n_ab_bonds = self.compute_respective_bond_counts(particle)
        coordinated_atoms = [len(particle.get_atom_indices_from_coordination_number([cn], symbol=self.symbol_a)) for cn in range(13)]

        M = particle.get_stoichiometry()[self.symbol_a]*0.1

        feature_vector = np.array([n_aa_bonds/n_atoms, n_bb_bonds/n_atoms, n_ab_bonds/n_atoms, M] + coordinated_atoms)
        particle.set_feature_vector(self.feature_key, feature_vector)


class TopologicalFeatureClassifier(SimpleFeatureClassifier):
    """Classifier for a generalization of the topological descriptors by Kozlov et al. (2015).

    Implemented for TWO elements, which are sorted alphabetically.The returned feature vector will have the form
    [n_aa_bonds/n_atoms, n_ab_bonds/n_atoms, n_bb_bonds/n_atoms, n_a_atoms*0.1, n_a(cn=0), n_a(cn=1), ..., n_a(cn=12].
    """
    def __init__(self, symbols):
        SimpleFeatureClassifier.__init__(self, symbols)
        self.feature_key = 'TFC'
        self.bond_scaling_factor = 1

    def compute_feature_vector(self, particle):
        n_atoms = particle.get_n_atoms()
        n_aa_bonds, n_bb_bonds, n_ab_bonds = self.compute_respective_bond_counts(particle)
        
        coordinated_atoms_a = [len(particle.get_atom_indices_from_coordination_number([cn], symbol=self.symbol_a)) for cn in range(13)]
        coordinated_atoms_b = [len(particle.get_atom_indices_from_coordination_number([cn], symbol=self.symbol_b)) for cn in range(13)]

        feature_vector = np.array([n_aa_bonds*self.bond_scaling_factor, n_bb_bonds*self.bond_scaling_factor,
                                   n_ab_bonds*self.bond_scaling_factor] + coordinated_atoms_a + coordinated_atoms_b)
        
        particle.set_feature_vector(self.feature_key, feature_vector)


class TopologicalEnvironmentFeatureClassifier(SimpleFeatureClassifier):
    def __init__(self, symbols):
        SimpleFeatureClassifier.__init__(self, symbols)
        symbols_copy = copy.deepcopy(symbols)
        self.symbols = sorted(symbols_copy)
        self.feature_key = 'TEC'
        
        self.coordination_number_offsets = [int(cn*(cn + 1)/2) for cn in range(13)]
    
    def compute_n_features(self, particle):
        return 182
    
    def compute_atom_feature(self, atom_index, particle):
        symbol = particle.get_symbol(atom_index)
        symbol_index = self.symbols.index(symbol)

        element_offset = symbol_index*91

        coordination_number = len(particle.neighbor_list[atom_index])
        symbols = [particle.get_symbol(neigh_index) for neigh_index in particle.neighbor_list[atom_index]] 
        if symbol == self.symbol_a:
            n_ab_bonds = symbols.count(self.symbol_b)
        else:
            n_ab_bonds = symbols.count(self.symbol_a)

        atoms_feature = int(self.coordination_number_offsets[coordination_number] + n_ab_bonds + element_offset)

        return atoms_feature

    def compute_feature_vector(self, particle):
        feature_vector = np.zeros(self.compute_n_features(particle))

        for atom_index in particle.get_indices():
            atom_feature = self.compute_atom_feature(atom_index, particle)
            feature_vector[atom_feature] += 1

        particle.set_feature_vector(self.feature_key, feature_vector)


# TODO move to separate modules & rename
class CoordinationFeatureClassifier(SimpleFeatureClassifier):
    """Feature classifier that counts how often each coordination number appears if vacancies (atoms of element 'X')
     are present in an otherwise PURE particle.

     Form of the feature vector:
     [n_aa_bonds, n_a(cn=0), n_a(cn=1), ... n_a(cn=12)]
     """
    def __init__(self, symbols):
        SimpleFeatureClassifier.__init__(self, symbols)
        self.feature_key = 'TFC'

    def compute_feature_vector(self, particle):
        n_aa_bonds, n_bb_bonds, n_ab_bonds = self.compute_respective_bond_counts(particle)
        symbol_a_features = [0]*13
        for index in particle.get_indices():
            if particle.get_symbol(index) != 'X':
                env = particle.get_local_environment(index)
                symbol_a_features[env[0]] += 1

        features = [n_aa_bonds] + symbol_a_features
        feature_vector = np.array(features)
        particle.set_feature_vector(self.feature_key, feature_vector)


class DipoleMomentCalculator(SimpleFeatureClassifier):
    
    def __init__(self, symbols):
        SimpleFeatureClassifier.__init__(self, symbols)
        self.feature_key = 'MU'

    def compute_feature_vector(self, particle, charges = [1, -1]):
    
        symbols = particle.get_all_symbols()
        fake_charges = {symbols[0] : charges[0], symbols[1] : charges[1]}
        partial_charges = [fake_charges[symbol] for symbol in particle.get_symbols()]

        dipole_moments = []
        environments = []
        for central_atom_idx in particle.get_atom_indices_from_coordination_number([12]):
            particle.translate_atoms_positions(particle.get_position(central_atom_idx))
            dipole_moment = 0
            for atom_idx in particle.get_coordination_atoms(central_atom_idx):
                dipole_moment += partial_charges[atom_idx] * particle.get_position(atom_idx)

            dipole_moments.append(np.linalg.norm(dipole_moment))
            environments.append(particle.get_coordination_atoms(central_atom_idx))

        feature_vector = np.average(dipole_moments)#/particle.get_n_atoms()

        particle.set_feature_vector(self.feature_key, feature_vector)

class AdsorptionFeatureVector(SimpleFeatureClassifier):

    def __init__(self, symbols):
        SimpleFeatureClassifier.__init__(self, symbols)
        self.feature_key = 'ADS'
        self.features_type = defaultdict(lambda : 0)
        self.n_features = 0

        self.get_features(symbols)


    def compute_feature_vector(self, particle):
        feature_vector = np.array([0 for _ in range(self.n_features)])
        for site_occupied in particle.get_occupation_status_by_indices(1):
            site_type = sorted(particle.get_symbols(list(particle.get_site_atom_indices(site_occupied))))
            site_type = tuple(site_type)
            index = self.features_type[site_type]
            feature_vector[index] += 1

        particle.set_feature_vector(self.feature_key, feature_vector)

    def get_features(self, symbols):
        symbols = sorted(symbols)
        index = 0
        for number_of_atom_in_site in range(1,4):
            for site_type in itertools.combinations_with_replacement(symbols,number_of_atom_in_site):
                self.features_type[site_type] = index
                index += 1
        self.n_features = len(self.features_type)

class GeneralizedCoordinationNumber(SimpleFeatureClassifier):

    """
    Class that only works for Monometallic Nanopaticles
    """

    def __init__(self, particle):

        particle.construct_adsorption_list()
        self.ads_list = particle.get_adsorption_as_list()
        self.all_gcn = list(particle.get_generalized_coordination_numbers(self.ads_list).keys())
        self.n_features = len(self.all_gcn)
        self.gcn_site_list = np.zeros(particle.get_total_number_of_sites())
        self.feature_key = 'GCN'
    
        for i, site_atom_indices in enumerate(self.ads_list):
            self.gcn_site_list[i] = particle.get_generalized_coordination_number(site_atom_indices)

    def compute_feature_vector(self, particle):
        feature_vector = np.zeros(self.n_features)
        occupied_site_indices = particle.get_indices_of_adsorbates()
        print(occupied_site_indices)
        for occupied_site_index in occupied_site_indices:
            index = self.all_gcn.index(self.gcn_site_list[occupied_site_index])
            feature_vector[index] += 1  

        particle.set_feature_vector(self.feature_key, feature_vector)
        







