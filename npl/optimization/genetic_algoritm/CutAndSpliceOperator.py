from Core.Nanoparticle import Nanoparticle
from Core.CuttingPlaneUtilities import SphericalCuttingPlaneGenerator
import numpy as np


class CutAndSpliceOperator:
    def __init__(self, max_radius, recompute_neighbor_list=True, normal_dir=None):
        self.cutting_plane_generator = SphericalCuttingPlaneGenerator(max_radius, 0.0, 0.0, normal_dir)
        self.recompute_neighbor_list = recompute_neighbor_list

    def cut_and_splice(self, p1, p2, fixed_stoichiometry=True):
        self.cutting_plane_generator.set_center(p1.get_ase_atoms().get_center_of_mass())
        self.cutting_plane_generator.set_max_radius(np.max(p1.get_ase_atoms().get_positions() -
                                                           p1.get_ase_atoms().get_center_of_mass()))
        # Ensure that we indeed cut, i.e. take atoms from both particles
        while True:
            cutting_plane = self.cutting_plane_generator.generate_new_cutting_plane()
            atom_indices_in_positive_subspace, _ = cutting_plane.split_atom_indices(p1.get_ase_atoms())
            _, atom_indices_in_negative_subspace = cutting_plane.split_atom_indices(p2.get_ase_atoms())

            if np.sum(atom_indices_in_negative_subspace) > 0 and np.sum(atom_indices_in_positive_subspace) > 0:
                break

        new_particle = Nanoparticle()
        new_particle.add_atoms(p1.get_ase_atoms().copy()[atom_indices_in_positive_subspace], False)
        new_particle.add_atoms(p2.get_ase_atoms().copy()[atom_indices_in_negative_subspace], False)

        if fixed_stoichiometry is True:
            target_stoichiometry = p1.get_stoichiometry()
            new_particle.adjust_stoichiometry(target_stoichiometry)

        if self.recompute_neighbor_list:
            new_particle.construct_neighbor_list()

        return new_particle
