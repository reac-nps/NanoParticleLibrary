from collections import defaultdict

import numpy as np
import itertools
import copy

from ase.neighborlist import natural_cutoffs
from ase.neighborlist import build_neighbor_list


class NeighborList:
    def __init__(self):
        self.list = defaultdict(lambda: set())

    def __getitem__(self, item):
        return self.list[item]

    def __setitem__(self, key, value):
        self.list[key] = value

    def construct(self, atoms, scale_factor=1.0, npl = True, cutoffs = None):

        if not cutoffs:
            cutoffs = natural_cutoffs(atoms, mult=scale_factor)
            
        neighbor_list = build_neighbor_list(atoms,
                                            cutoffs=cutoffs,
                                            bothways=True,
                                            self_interaction=False)

        for atom_idx, _ in enumerate(atoms):
            neighbors, _ = neighbor_list.get_neighbors(atom_idx)
            if npl:
                self.list[atom_idx] = set(neighbors)
            else:
                self.list[atom_idx] = neighbors

    def get_coordination_number(self, atom_idx):
        return len(self.list[atom_idx])

    def get_n_bonds(self):
        n_bonds = sum([len(l) for l in list(self.list.values())])
        return n_bonds/2

    def get_coordination_atoms(self, atom_idx):
        return list(self.list[atom_idx])
    
    def get_max_coordination_number(self, indices):
        if len(indices) == 1:
            return 12
        if len(indices) == 2:
            return 18
        if len(indices) == 3:
            return 22
        if len(indices) == 4:
            return 26

        common_atom_indices = []
        for pair in itertools.combinations(indices, 2):
            shared_pair = list(np.intersect1d(self.get_coordination_atoms(pair[0]), self.get_coordination_atoms(pair[1])))
            common_atom_indices += shared_pair    

        return 12*len(indices) - len(set(common_atom_indices))
    
    def get_generalized_coordination_number(self, indices):
        max_coordination = self.get_max_coordination_number(indices)
        coordination_atoms = []
        for atom_idx in indices:
            for neigh in self.get_coordination_atoms(atom_idx):
                if neigh not in coordination_atoms and neigh not in indices:
                    coordination_atoms.append(neigh)
       
        tot_cns = 0
        for neighbor_idx in coordination_atoms:
            cn = self.get_coordination_number(neighbor_idx)
            tot_cns += cn
        
        return round(tot_cns / max_coordination, 2)
    
    def get_atoms_in_the_surface_plane(self, atom_idx, edges_corner=False):
        """Find atoms within the same surface, excluding bulk atoms,
        atom must be in a terrace position. To include edges and cornes edges_corners=True"""
        cn = self.get_coordination_number(atom_idx)
        if cn != 9 and cn != 8:
            raise Exception('AtomIsNotInTerrace')

        atoms_indices_in_plane = [atom_idx]
        for atom_idx in atoms_indices_in_plane:
            neighbors = self.get_coordination_atoms(atom_idx)
            for neighbor in neighbors:
                cn_neighbor = self.get_coordination_number(neighbor)
                if cn_neighbor == cn and neighbor not in atoms_indices_in_plane:
                    atoms_indices_in_plane.append(neighbor)
        
        if edges_corner == True:
            for atom_idx in atoms_indices_in_plane:
                neighbors = self.get_coordination_atoms(atom_idx) 
                for neighbor in neighbors:
                    cn_neighbor = self.get_coordination_number(neighbor)
                    if cn_neighbor < cn and neighbor not in atoms_indices_in_plane:
                        atoms_indices_in_plane.append(neighbor)

        return atoms_indices_in_plane
        




