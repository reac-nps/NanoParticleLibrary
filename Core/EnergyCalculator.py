from ase.optimize import BFGS
from asap3 import EMT

import numpy as np
import copy
import sklearn.gaussian_process as gp
from sklearn.linear_model import BayesianRidge


class EnergyCalculator:
    """Base class for an energy calculator.

    Valid implementations have to implement the compute_energy(particle) function.
    Energies are saved in the particle object with the key of the respective calculator.
    """
    def __init__(self):
        self.energy_key = None
        pass

    def compute_energy(self, particle):
        raise NotImplementedError

    def get_energy_key(self):
        return copy.deepcopy(self.energy_key)

    def set_energy_key(self, energy_key):
        self.energy_key = energy_key


class EMTCalculator(EnergyCalculator):
    """Wrapper class around the asap3 EMT calculator."""

    def __init__(self, fmax=0.01, steps=20):
        EnergyCalculator.__init__(self)
        self.fmax = fmax
        self.steps = steps
        self.energy_key = 'EMT'

    def compute_energy(self, particle, relax_atoms=False):
        """Compute the energy using EMT.

        BFGS is used for relaxation. By default, the atoms are NOT relaxed, i.e. the
        geometry remains unchanged unless this is explicitly stated.

        Parameters:
            particle : Nanoparticle
            relax_atoms : bool
        """
        cell_width = 1e3
        cell_height = 1e3
        cell_length = 1e3

        atoms = particle.get_ase_atoms(exclude_x=True)
        if not relax_atoms:
            atoms = atoms.copy()

        atoms.set_cell(np.array([[cell_width, 0, 0], [0, cell_length, 0], [0, 0, cell_height]]))
        atoms.set_calculator(EMT())
        dyn = BFGS(atoms)
        dyn.run(fmax=self.fmax, steps=self.steps)

        energy = atoms.get_potential_energy()
        particle.set_energy(self.energy_key, energy)


class GPRCalculator(EnergyCalculator):
    """Energy calculator using global feature vectors and Gaussian Process Regression."""

    def __init__(self, feature_key, kernel=None, alpha=0.01, normalize_y=True):
        EnergyCalculator.__init__(self)
        if kernel is None:
            self.kernel = gp.kernels.ConstantKernel(1., (1e-1, 1e3)) * gp.kernels.RBF(1., (1e-3, 1e3))
        else:
            self.kernel = kernel

        self.alpha = alpha
        self.normalize_y = normalize_y
        self.GPR = None
        self.energy_key = 'GPR'
        self.feature_key = feature_key

    def fit(self, training_set, energy_key):
        """Fit the GPR model.

        The feature vectors with key = self.feature_key will be used for feature vectors. The
        energy with the specified energy_key will be the target function.

        Parameters:
            training_set : list of Nanoparticles
            energy_key : str
        """
        feature_vectors = [p.get_feature_vector(self.feature_key) for p in training_set]
        energies = [p.get_energy(energy_key) for p in training_set]

        self.GPR = gp.GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=20, alpha=self.alpha,
                                               normalize_y=self.normalize_y)
        self.GPR.fit(feature_vectors, energies)

    def compute_energy(self, particle):
        """Compute the energy using GPR.

        Assumes that a feature vector with key=self.feature_key is present in the particle.

        Parameters:
            particle : Nanoparticle
        """
        energy = self.GPR.predict([particle.get_feature_vector(self.feature_key)])[0]
        particle.set_energy(self.energy_key, energy)


class MixingEnergyCalculator(EnergyCalculator):
    """Compute the mixing energy using an arbitrary energy model.

    For the original energy model it is assumed that all previous steps in the energy pipeline, e.g. calculation
    of local environment, feature vectors etc. has been carried out.
    """
    def __init__(self, base_calculator=None, mixing_parameters=None, recompute_energies=False):
        EnergyCalculator.__init__(self)

        if mixing_parameters is None:
            self.mixing_parameters = dict()
        else:
            self.mixing_parameters = mixing_parameters

        self.base_calculator = base_calculator
        self.recompute_energies = recompute_energies
        self.energy_key = 'Mixing Energy'

    def compute_mixing_parameters(self, particle, symbols):
        """Compute the energies for the pure particles of the given symbols as reference points.

        Parameters:
            particle : Nanoparticle
            symbols : list of str
        """
        for symbol in symbols:
            particle.random_ordering({symbol: 1.0})
            self.base_calculator.compute_energy(particle)
            self.mixing_parameters[symbol] = particle.get_energy('EMT')

    def compute_energy(self, particle):
        """Compute the mixing energy of the particle using the base energy model.

        If energies have been computed using the same energy model as the base calculator they are reused if
        self.recompute_energies == False

        Parameters:
            particle : Nanoparticle
        """
        if self.recompute_energies:
            self.base_calculator.compute_energy(particle)

        energy_key = self.base_calculator.get_energy_key()
        mixing_energy = particle.get_energy(energy_key)

        n_atoms = particle.atoms.get_n_atoms()

        for symbol in particle.get_stoichiometry():
            mixing_energy -= self.mixing_parameters[symbol] * particle.get_stoichiometry()[symbol] / n_atoms

        particle.set_energy(self.energy_key, mixing_energy)


class BayesianRRCalculator(EnergyCalculator):
    """Energy calculator using global feature vectors and Bayesian Ridge Regression."""

    def __init__(self, feature_key):
        EnergyCalculator.__init__(self)

        self.ridge = BayesianRidge(fit_intercept=False)
        self.energy_key = 'BRR'
        self.feature_key = feature_key

    def fit(self, training_set, energy_key):
        """Fit the BRR model.

        The feature vectors with key=self.feature_key will be used for feature vectors. The
        energy with the specified energy_key will be the target function.

        Parameters:
            training_set : list of Nanoparticles
            energy_key : str
        """
        feature_vectors = [p.get_feature_vector(self.feature_key) for p in training_set]
        energies = [p.get_energy(energy_key) for p in training_set]

        self.ridge.fit(feature_vectors, energies)

    def get_coefficients(self):
        return self.ridge.coef_

    def set_coefficients(self, new_coefficients):
        self.ridge.coef_ = new_coefficients

    def set_feature_key(self, feature_key):
        self.feature_key = feature_key

    def compute_energy(self, particle):
        """Compute the energy using BRR.

        Assumes that a feature vector with key=self.feature_key is present in the particle.

        Parameters:
            particle : Nanoparticle
        """
        brr_energy = np.dot(np.transpose(self.ridge.coef_), particle.get_feature_vector(self.feature_key))
        particle.set_energy(self.energy_key, brr_energy)


class DipoleMomentCalculator:
    
    def __init__(self):
        self.total_dipole_moment = None
        self.dipole_moments = None
        self.environments = None
        pass

    def compute_dipole_moment(self, particle, charges = [1, -1]):
    
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

        self.total_dipole_moment = np.average(dipole_moments)/particle.get_n_atoms()
        self.dipole_moments = dipole_moments
        self.environments = environments

    def get_total_dipole_moment(self):
        return self.total_dipole_moment
    
    def get_dipole_moments(self):
        return self.dipole_moments

    def get_environments(self):
        return self.environments

        

# TODO move to relevant file -> Basin Hopping, Local optimization
# TODO remove scaling factors from topological descriptors
def compute_coefficients_for_linear_topological_model(global_topological_coefficients, symbols, n_atoms):
    coordination_numbers = list(range(13))
    symbols_copy = copy.deepcopy(symbols)
    symbols_copy.sort()
    symbol_a = symbols_copy[0]
    print("Coef symbol_a: {}".format(symbol_a))

    e_aa_bond = global_topological_coefficients[0]/n_atoms
    e_bb_bond = global_topological_coefficients[1]/n_atoms
    e_ab_bond = global_topological_coefficients[2]/n_atoms

    coefficients = []
    total_energies = []
    for symbol in symbols_copy:
        for cn_number in coordination_numbers:
            for n_symbol_a_atoms in range(cn_number + 1):
                energy = 0

                if symbol == symbol_a:
                    energy += (global_topological_coefficients[3]*0.1)  # careful...
                    energy += (n_symbol_a_atoms*e_aa_bond/2)
                    energy += ((cn_number - n_symbol_a_atoms)*e_ab_bond/2)
                    energy += (global_topological_coefficients[4 + cn_number])

                    total_energy = energy
                    total_energy += n_symbol_a_atoms*e_aa_bond/2
                    total_energy += (cn_number - n_symbol_a_atoms)*e_ab_bond/2
                else:
                    energy += (n_symbol_a_atoms*e_ab_bond/2)
                    energy += ((cn_number - n_symbol_a_atoms)*e_bb_bond/2)

                    total_energy = energy
                    total_energy += n_symbol_a_atoms*e_ab_bond/2
                    total_energy += (cn_number - n_symbol_a_atoms)*e_bb_bond/2

                coefficients.append(energy)
                total_energies.append(total_energy)

    coefficients = np.array(coefficients)

    return coefficients, total_energies


def compute_coefficients_for_shape_optimization(global_topological_coefficients, symbols):
    coordination_numbers = list(range(13))
    symbols_copy = copy.deepcopy(symbols)
    symbols_copy.sort()

    e_aa_bond = global_topological_coefficients[0]

    coordination_energies_a = dict()
    for index, cn in enumerate(coordination_numbers):
        coordination_energies_a[cn] = global_topological_coefficients[index + 1]

    coefficients = []
    total_energies = []
    for cn_number in coordination_numbers:
        for n_symbol_a_atoms in range(cn_number + 1):
            energy = 0

            energy += (n_symbol_a_atoms*e_aa_bond/2)
            energy += (coordination_energies_a[cn_number])

            total_energy = energy + n_symbol_a_atoms*e_aa_bond/2

            coefficients.append(energy)
            total_energies.append(total_energy)

    coefficients += [0]*len(coefficients)
    total_energies += [0]*len(total_energies)

    coefficients = np.array(coefficients)

    return coefficients, total_energies
