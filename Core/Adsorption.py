import Core.MathModules as math

from ase import Atoms

import copy
import itertools
import numpy as np

from collections import defaultdict

class AdsorptionSiteList():

    def __init__(self):
        self.list = defaultdict(lambda : set())
        self.total_n_sites = 0
        self.occupation_vector = []

    def __getitem__(self, item):
        return self.list[item]

    def __setitem__(self, key, value):
        self.list[key] = value

    def construct(self, particle):
        adsorption_site_list = self.build_site_list(particle)
        for site_index, atom_indices in enumerate(adsorption_site_list):
            self.list[site_index] = set(atom_indices)

        self.total_n_sites = len(self.list)
        self.occupation_vector = np.array([0 for _ in range(self.total_n_sites)])

    def random_occupation(self, number_of_adsorbates):
        # Reset the occupation vector
        self.occupation_vector = np.array([0 for _ in range(self.total_n_sites)])

        occupied_sites_indices = np.random.choice(np.arange(self.total_n_sites), number_of_adsorbates, replace=False)
        for site_index in occupied_sites_indices:
                self.occupation_vector[site_index] = 1 

    def get_total_number_of_sites(self):
        return self.total_n_sites

    def get_number_of_adsorbates(self):
        return len(self.get_occupation_status_by_indices(1))

    def get_site_atom_indices(self, index):
        return self.list[index]

    def get_occupation_status_by_indices(self, status):
        # status : occupied (1) or unoccupied (0) adsorption site
        return np.where(self.occupation_vector == status)[0]

    def occupation_vector(self):
        return self.occupation_vector

    def swap_status(self, index_pairs):
        for idx1, idx2 in index_pairs:
            self.occupation_vector[idx1], self.occupation_vector[idx2] = self.occupation_vector[idx2], self.occupation_vector[idx1]

    def build_site_list(self, particle):
        def find_plane_for_bridge_atoms(particle, indices):
            uncoordinated_atoms = set(particle.get_atom_indices_from_coordination_number(range(12)))
            shell_1 = set(particle.get_coordination_atoms(indices[0]))
            shell_2 = set(particle.get_coordination_atoms(indices[1]))
            shared_atoms = uncoordinated_atoms.intersection(shell_1.intersection(shell_2))
            return list(shared_atoms)

        atoms_in_surface = set(particle.get_atom_indices_from_coordination_number(range(10)))

        ontop_sites = []
        for atom in atoms_in_surface:
            ontop_sites.append([atom])

        bridge_sites = []
        for central_atom_index in atoms_in_surface:
            central_atom_nearest_neighbors = set(particle.get_coordination_atoms(central_atom_index))
            for nearest_neighbor in atoms_in_surface.intersection(central_atom_nearest_neighbors):
                pair = sorted([central_atom_index, nearest_neighbor])
                if pair not in bridge_sites:
                    bridge_sites.append(pair)

        hollow_sites = []
        for pair in bridge_sites:
            for third_atom in find_plane_for_bridge_atoms(particle, pair):
                triplet = copy.copy(pair)
                triplet.append(third_atom)
                triplet = sorted(triplet)
                if triplet not in hollow_sites:
                    hollow_sites.append(triplet)

        return ontop_sites + bridge_sites + hollow_sites


        
class FindAdsorptionSites():
    """ Class that identify and place add atoms based on the Generalized coordination Numbers of the nanoparticles"""
    def __init__(self):
        self.ontop = []
        self.bridge_positions = []
        self.hollow_positions = []
    
    def get_ontop_sites(self, particle):
        for atom in particle.get_atom_indices_from_coordination_number(range(10)):
            self.ontop.append([atom])
            
    def get_bridge_sites(self, particle):
        shell_atoms = set(particle.get_atom_indices_from_coordination_number(range(10)))
        for central_atom_index in shell_atoms:
            central_atom_nearest_neighbors = set(particle.get_coordination_atoms(central_atom_index))
            for nearest_neighbor in shell_atoms.intersection(central_atom_nearest_neighbors):
                pair = sorted([central_atom_index, nearest_neighbor])
                if pair not in self.bridge_positions:
                    self.bridge_positions.append(pair)
                    
    def get_hollow_sites(self, particle):      
        for pair in self.bridge_positions:
            for third_atom in self.find_plane_for_bridge_atoms(particle, pair):
                triplet = copy.copy(pair)
                triplet.append(third_atom)
                triplet = sorted(triplet)
                if triplet not in self.hollow_positions:
                    self.hollow_positions.append(triplet)
    
    def find_plane_for_bridge_atoms(self, particle, indices):
        uncoordinated_atoms = particle.get_atom_indices_from_coordination_number(range(12))
        uncoordinated_atoms = set(uncoordinated_atoms)
        shell_1 = set(particle.get_coordination_atoms(indices[0]))
        shell_2 = set(particle.get_coordination_atoms(indices[1]))
        shared_atoms = uncoordinated_atoms.intersection(shell_1.intersection(shell_2))
        return list(shared_atoms)
    
class PlaceAddAtoms():
    """Class that plance add atoms on positions identified by FindAdsorptionSites"""
    def __init__(self, symbols):
        self.adsorption_sites = FindAdsorptionSites()
        self.symbols = sorted(symbols)
        self.sites_list = []
        self.ontop_positions = {site[0] : [] for site in itertools.combinations_with_replacement(self.symbols,1)}
        self.bridge_positions = {''.join(list(site)) : [] for site in itertools.combinations_with_replacement(self.symbols,2)}
        self.hollow_positions = {''.join(list(site)) : [] for site in itertools.combinations_with_replacement(self.symbols,3)}

    def bind_particle(self, particle):
        particle.atoms.atoms.translate(-particle.atoms.atoms.get_center_of_mass())
        self.adsorption_sites.get_ontop_sites(particle)
        self.adsorption_sites.get_bridge_sites(particle)
        self.adsorption_sites.get_hollow_sites(particle)  
        self.get_total_adsorption_sites(particle)
        self.get_ontop_sites(particle)
        self.get_bridge_sties(particle)
        self.get_hollow_sties(particle)
    
    def get_ontop_sites(self, particle):
        for atom in self.adsorption_sites.ontop:
            self.ontop_positions[particle.get_symbol(atom[0])].append(atom[0])

    def get_bridge_sties(self, particle):
        for pairs in self.adsorption_sites.bridge_positions:
            self.bridge_positions[''.join(sorted(particle.get_symbols(pairs)))].append(pairs)
 
    def get_hollow_sties(self, particle):
        for triplet in self.adsorption_sites.hollow_positions:
            self.hollow_positions[''.join(sorted(particle.get_symbols(triplet)))].append(triplet)
            
    def get_total_adsorption_sites(self, particle):
        #self.sites_list += self.adsorption_sites.ontop
        #self.sites_list += self.adsorption_sites.bridge_positions
        self.sites_list += self.adsorption_sites.hollow_positions

    def get_xyz_site_from_atom_indices(self, particle, site):
        
        #if isinstance(site, (np.ndarray, np.generic)) or isinstance(site, int):
        if len(site) == 1:
            unit_vector1, length1 = math.get_unit_vector(particle.get_position(site[0]))
            xyz_site = (unit_vector1*(length1+2))
            return xyz_site
        else:
            xyz_atoms = [particle.get_position(atom_index) for atom_index in site]
            xyz_site_plane = math.find_middle_point(xyz_atoms)
            unit_vector1, length1 = math.get_unit_vector(xyz_site_plane)
            if len(site) == 3:
                normal_vector = math.get_normal_vector(xyz_atoms) 
            if len(site) == 2:
                third_atom = self.adsorption_sites.find_plane_for_bridge_atoms(particle, site)
                if isinstance(third_atom, int):
                    xyz_third_atom = particle.get_position(third_atom)
                else:
                    xyz_third_atom = particle.get_position(third_atom[0])
                xyz_atoms.append(xyz_third_atom)
                normal_vector = math.get_normal_vector(xyz_atoms)
                
            unit_vector2, length2 = math.get_unit_vector(normal_vector)
            dot_prod = np.dot(unit_vector1, unit_vector2)
            direction = dot_prod/abs(dot_prod)
            xyz_site = (unit_vector1*(length1))+(direction*unit_vector2*(1.4))
   
        return xyz_site

    def place_add_atom(self, particle, add_atom_symbol, sites):
        add_atom_list = []
        for site in sites:
            xyz_site = self.get_xyz_site_from_atom_indices(particle, site)
            add_atom = Atoms(add_atom_symbol)   
            add_atom.translate(xyz_site)
            add_atom_list.append(add_atom)
            
        for add_atom in add_atom_list:
            particle.add_atoms(add_atom)
            
        return particle    





