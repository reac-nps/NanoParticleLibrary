from LocalOpt.LocalOptimization import local_optimization

import copy


class LocalOptimizationOperator:
    def __init__(self, energy_calculator, environment_energies):
        self.energy_calculator = energy_calculator
        self.environment_energies = environment_energies

    def local_optimization(self, particle):
        new_particle = copy.deepcopy(particle)
        new_particle, accepted_energies = local_optimization(new_particle,
                                                             self.energy_calculator,
                                                             self.energy_calculator)

        return new_particle, accepted_energies
