import numpy as np

from npl.core.base_nanoparticle import BaseNanoparticle
from npl.core.adsorption import FindAdsorptionSites

from ase.cluster.octahedron import Octahedron
from ase import Atoms


class Nanoparticle(BaseNanoparticle):
    def __init__(self):
        BaseNanoparticle.__init__(self)
        self.adsorption = FindAdsorptionSites()

    def truncated_octahedron(self, height, cutoff, stoichiometry, lattice_constant=3.9, alloy=False):
        octa = Octahedron('Pt', height, cutoff, latticeconstant=lattice_constant, alloy=alloy)
        atoms = Atoms(octa.symbols, octa.positions)
        com = atoms.get_center_of_mass()
        atoms.positions -= com

        self.add_atoms(atoms, recompute_neighbor_list=False)
        self.random_ordering(stoichiometry)
        self.construct_neighbor_list()

    def adjust_stoichiometry(self, target_stoichiometry):
        def choose_n_atoms_with_symbol(symbol, n_atoms):
            atoms_with_symbol = self.get_indices_by_symbol(symbol)
            selected_atoms = np.random.choice(atoms_with_symbol, n_atoms, replace=False)
            return selected_atoms

        excess_atoms = np.array([])
        for symbol in self.get_stoichiometry():
            if symbol in target_stoichiometry:
                difference = self.get_stoichiometry()[symbol] - target_stoichiometry[symbol]
            else:
                difference = self.get_stoichiometry()[symbol]

            if difference > 0:
                excess_atoms = np.append(excess_atoms, choose_n_atoms_with_symbol(symbol, difference), 0)

        np.random.shuffle(excess_atoms)
        excess_atoms = excess_atoms.astype(np.int)

        for symbol in target_stoichiometry:
            difference = target_stoichiometry[symbol]
            if symbol in self.get_stoichiometry():
                difference = target_stoichiometry[symbol] - self.get_stoichiometry()[symbol]
            self.transform_atoms(excess_atoms[-difference:], [symbol]*difference)
            excess_atoms = excess_atoms[:-difference]
        return



