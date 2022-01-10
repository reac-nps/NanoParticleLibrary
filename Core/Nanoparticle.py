import numpy as np

from Core.BaseNanoparticle import BaseNanoparticle
from Core.Adsorption import FindAdsorptionSites

from ase.cluster import Octahedron
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

    def get_adsoption_site_ocupation(self, random_vector = None):
        # random_vector = dict() 
        self.adsorption_site_list.construct(self)
        total_adsorption_sites = self.adsorption_site_list.get_total_number_of_adsorption_sites()
        occupational_vector = [0 for _ in range(total_adsorption_sites)]

        if random_vector is not None:
            n_adsorbates = random_vector['n_adsorbates']
            occupied_sites_indices = np.random.choice(np.arange(total_adsorption_sites), n_adsorbates, replace=False)
            for site_index in occupied_sites_indices:
                occupational_vector[site_index] = 1
            return occupational_vector
        else:
            return occupational_vector



