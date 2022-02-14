import numpy as np
import copy
from itertools import chain

from Core.LocalEnvironmentCalculator import NeighborCountingEnvironmentCalculator
from MCMC.RandomExchangeOperator import RandomExchangeOperator


def setup_monte_carlo(start_particle, energy_calculator, local_feature_classifier):
    symbols = start_particle.get_all_symbols()
    energy_key = energy_calculator.get_energy_key()

    local_env_calculator = NeighborCountingEnvironmentCalculator(symbols)
    local_env_calculator.compute_local_environments(start_particle)

    local_feature_classifier.compute_feature_vector(start_particle)
    energy_calculator.compute_energy(start_particle)

    exchange_operator = RandomExchangeOperator(0.5)
    exchange_operator.bind_particle(start_particle)

    return energy_key, local_env_calculator, exchange_operator


def update_atomic_features(exchanges, local_env_calculator, local_feature_classifier, particle):
    neighborhood = set()
    for exchange in exchanges:
        index1, index2 = exchange
        neighborhood.add(index1)
        neighborhood.add(index2)

        neighborhood = neighborhood.union(particle.neighbor_list[index1])
        neighborhood = neighborhood.union(particle.neighbor_list[index2])

    for index in neighborhood:
        local_env_calculator.compute_local_environment(particle, index)
        local_feature_classifier.compute_atom_feature(particle, index)

    local_feature_classifier.compute_feature_vector(particle, recompute_atom_features=False)
    return particle, neighborhood

def run_monte_carlo(beta, max_steps, start_particle, energy_calculator, local_feature_classifier):
    energy_key, local_env_calculator, exchange_operator = setup_monte_carlo(start_particle, energy_calculator,
                                                                            local_feature_classifier)

    start_energy = start_particle.get_energy(energy_key)
    lowest_energy = start_energy
    accepted_energies = [(lowest_energy, 0)]

    found_new_solution = False
    fields = ['energies', 'symbols']
    best_particle = copy.deepcopy(start_particle.get_as_dictionary(fields))

    total_steps = 0
    no_improvement = 0
    while no_improvement < max_steps:
        total_steps += 1
        if total_steps % 2000 == 0:
            print("Step: {}".format(total_steps))
            print("Lowest energy: {}".format(lowest_energy))

        exchanges = exchange_operator.random_exchange(start_particle)

        start_particle, neighborhood = update_atomic_features(exchanges, local_env_calculator, local_feature_classifier,
                                                              start_particle)

        energy_calculator.compute_energy(start_particle)
        new_energy = start_particle.get_energy(energy_key)

        delta_e = new_energy - start_energy

        acceptance_rate = min(1, np.exp(-beta * delta_e))
        if np.random.random() < acceptance_rate:
            if found_new_solution:
                if new_energy > start_energy:
                    start_particle.swap_symbols(exchanges)
                    best_particle = copy.deepcopy(start_particle.get_as_dictionary(fields))
                    best_particle['energies'][energy_key] = start_energy
                    start_particle.swap_symbols(exchanges)

            start_energy = new_energy
            accepted_energies.append((new_energy, total_steps))

            if new_energy < lowest_energy:
                no_improvement = 0
                lowest_energy = new_energy
                found_new_solution = True
            else:
                no_improvement += 1
                found_new_solution = False

        else:
            no_improvement += 1

            # roll back exchanges and make sure features and environments are up-to-date
            start_particle.swap_symbols(exchanges)
            start_particle.set_energy(energy_key, start_energy)
            for index in neighborhood:
                local_env_calculator.compute_local_environment(start_particle, index)
                local_feature_classifier.compute_atom_feature(start_particle, index)

            if found_new_solution:
                best_particle = copy.deepcopy(start_particle.get_as_dictionary(fields))
                best_particle['energies'][energy_key] = copy.deepcopy(start_energy)
                found_new_solution = False

    accepted_energies.append((accepted_energies[-1][0], total_steps))

    return [best_particle, accepted_energies]

def run_monte_carlo_for_adsorbates(beta, max_steps, start_particle, adsorbates_energy, n_adsorbates):
    from Core.GlobalFeatureClassifier import AdsorptionFeatureVector
    from Core.EnergyCalculator import LateralInteractionCalculator

    exchange_operator = RandomExchangeOperator(0.5)
    exchange_operator.bind_adsorbates(start_particle, n_adsorbates)

    symbols = start_particle.get_all_symbols()
    adsorbates_energy_key = adsorbates_energy.get_energy_key()

    global_feature_classifier = AdsorptionFeatureVector(symbols)
    global_feature_classifier.compute_feature_vector(start_particle)
    adsorbates_energy.compute_energy(start_particle)

    lateral_interaction_calculator = LateralInteractionCalculator()
    lateral_interaction_energy_key = lateral_interaction_calculator.energy_key
    lateral_interaction_calculator.bind_grid(start_particle)
    lateral_interaction_calculator.compute_energy(start_particle)

    def get_ordering_and_adsorbates_energy(particle):
        adsorbates_energy = start_particle.get_energy(adsorbates_energy_key)
        lateral_interaction = start_particle.get_energy(lateral_interaction_energy_key)
        particle.set_energy('TOT', adsorbates_energy+lateral_interaction)
        return particle.get_energy('TOT')
    #energy_key, local_env_calculator, exchange_operator = setup_monte_carlo(start_particle, energy_calculator, local_feature_classifier)


    #initial_adsorbed_sites = start_particle.get_indices_of_adsorbates()
    

    start_energy = get_ordering_and_adsorbates_energy(start_particle)
    lowest_energy = start_energy
    sites_occupied = start_particle.get_indices_of_adsorbates()
    accepted_energies = [(lowest_energy, 0, sites_occupied)]

    found_new_solution = False
    fields = ['energies', 'symbols', 'positions']
    best_particle = copy.deepcopy(start_particle.get_as_dictionary(fields))

    total_steps = 0
    no_improvement = 0
    while no_improvement < max_steps:
        total_steps += 1
        if total_steps % 2000 == 0:
            print("Step: {}".format(total_steps))
            print("Lowest energy: {}".format(lowest_energy))

        exchanges = exchange_operator.random_adsorbate_migration(start_particle)

        accepted_particle = copy.deepcopy(start_particle)

        global_feature_classifier.compute_feature_vector(start_particle)
        adsorbates_energy.compute_energy(start_particle)
        lateral_interaction_calculator.compute_energy(start_particle)
        new_energy = get_ordering_and_adsorbates_energy(start_particle)


        delta_e = new_energy - start_energy

        acceptance_rate = min(1, np.exp(-beta * delta_e))
        if np.random.random() < acceptance_rate:
            if found_new_solution:
                if new_energy > start_energy:
                    start_particle.swap_status(exchanges)
                    best_particle = copy.deepcopy(start_particle.get_as_dictionary(fields))
                    best_particle['energies'][adsorbates_energy_key] = copy.deepcopy(start_energy)
                    best_particle['ads'] = copy.deepcopy(start_particle.get_indices_of_adsorbates())
                    best_particle['lateral_interaction'] = start_particle.get_energy(lateral_interaction_energy_key)
                    best_particle['adsorbates_energy'] = start_particle.get_energy(adsorbates_energy_key)
                    start_particle.swap_status(exchanges)

            sites_occupied = start_particle.get_indices_of_adsorbates()
            start_energy = new_energy
            accepted_energies.append((new_energy, total_steps, sites_occupied))

            if new_energy < lowest_energy:
                no_improvement = 0
                lowest_energy = new_energy
                found_new_solution = True
            else:
                no_improvement += 1
                found_new_solution = False

        else:
            no_improvement += 1

            # roll back exchanges and make sure features and environments are up-to-date
            start_particle.swap_status(exchanges)
            start_particle.set_energy('TOT', start_energy)

            if found_new_solution:
                best_particle = copy.deepcopy(start_particle.get_as_dictionary(fields))
                best_particle['energies'][adsorbates_energy_key] = copy.deepcopy(start_energy)
                best_particle['ads'] = copy.deepcopy(start_particle.get_indices_of_adsorbates())
                best_particle['lateral_interaction'] = start_particle.get_energy(lateral_interaction_energy_key)
                best_particle['adsorbates_energy'] = start_particle.get_energy(adsorbates_energy_key)
                found_new_solution = False

    accepted_energies.append((accepted_energies[-1][0], total_steps))
    return [best_particle, accepted_energies]


def run_monte_carlo_ordering_adsorbates(beta, max_steps, start_particle, ordering_energy_calculator, adsorbates_energy_calculator, n_adsorbates, local_feature_classifier):
    from Core.GlobalFeatureClassifier import AdsorptionFeatureVector
    from Core.EnergyCalculator import LateralInteractionCalculator

    exchange_operator = RandomExchangeOperator(0.5)
    exchange_operator.bind_adsorbates(start_particle, n_adsorbates)

    symbols = start_particle.get_all_symbols()
    adsorbates_energy_key = adsorbates_energy_calculator.get_energy_key()

    adsorbate_feature_classifier = AdsorptionFeatureVector(symbols)
    adsorbate_feature_classifier.compute_feature_vector(start_particle)
    adsorbates_energy_calculator.compute_energy(start_particle)

    ordering_energy_key, local_env_calculator, ordering_exchange_operator = setup_monte_carlo(start_particle, ordering_energy_calculator, local_feature_classifier)

    initial_adsorbed_sites = start_particle.get_indices_of_adsorbates()

    lateral_interaction_calculator = LateralInteractionCalculator()
    lateral_interaction_energy_key = lateral_interaction_calculator.energy_key
    lateral_interaction_calculator.bind_grid(start_particle)
    lateral_interaction_calculator.compute_energy(start_particle)

    def get_ordering_and_adsorbates_energy(particle):
        ordering_energy = start_particle.get_energy(ordering_energy_key)
        adsorbates_energy = start_particle.get_energy(adsorbates_energy_key)
        lateral_interaction = start_particle.get_energy(lateral_interaction_energy_key)
        particle.set_energy('TOT', ordering_energy + adsorbates_energy+lateral_interaction)
        return particle.get_energy('TOT')

    #start_energy = start_particle.get_energy(ordering_energy_key) + start_particle.get_energy(adsorbates_energy_key)
    start_energy = get_ordering_and_adsorbates_energy(start_particle)
    
    lowest_energy = start_energy
    sites_occupied = start_particle.get_indices_of_adsorbates()
    accepted_energies = [(lowest_energy, 0, sites_occupied)]

    found_new_solution = False
    fields = ['energies', 'symbols', 'positions']
    best_particle = copy.deepcopy(start_particle.get_as_dictionary(fields))
    best_particles = list()
    total_steps = 0
    no_improvement = 0
    while no_improvement < max_steps:
        total_steps += 1
        if total_steps % 2000 == 0:
            print("Step: {}".format(total_steps))
            print("Lowest energy: {}".format(lowest_energy))

        ordering_exchanges, adsorbates_exchanges = exchange_operator.coupled_random_exchange(start_particle)

        start_particle, neighborhood = update_atomic_features(ordering_exchanges, local_env_calculator, local_feature_classifier,
                                                              start_particle)

        accepted_particle = copy.deepcopy(start_particle)
        adsorbate_feature_classifier.compute_feature_vector(start_particle)

        ordering_energy_calculator.compute_energy(start_particle)
        adsorbates_energy_calculator.compute_energy(start_particle)
        lateral_interaction_calculator.compute_energy(start_particle)
        new_energy = get_ordering_and_adsorbates_energy(start_particle)

        delta_e = new_energy - start_energy

        acceptance_rate = min(1, np.exp(-beta * delta_e))
        if np.random.random() < acceptance_rate:
            if found_new_solution:
                if new_energy > start_energy:
                    start_particle.swap_symbols(ordering_exchanges)
                    start_particle.swap_status(adsorbates_exchanges)
                    best_particle = copy.deepcopy(start_particle.get_as_dictionary(fields))
                    best_particle['energies'][ordering_energy_key] = copy.deepcopy(start_energy)
                    best_particle['ads'] = copy.deepcopy(start_particle.get_indices_of_adsorbates())
                    best_particles.append(best_particle)
                    start_particle.swap_status(adsorbates_exchanges)
                    start_particle.swap_symbols(ordering_exchanges)

            sites_occupied = start_particle.get_indices_of_adsorbates()
            start_energy = new_energy
            accepted_energies.append((new_energy, total_steps, sites_occupied))

            if new_energy < lowest_energy:
                no_improvement = 0
                lowest_energy = new_energy
                found_new_solution = True
            else:
                no_improvement += 1
                found_new_solution = False

        else:
            no_improvement += 1

            # roll back exchanges and make sure features and environments are up-to-date
            start_particle.swap_status(adsorbates_exchanges)
            start_particle.swap_symbols(ordering_exchanges)
            start_particle.set_energy('TOT', start_energy)

            for index in neighborhood:
                local_env_calculator.compute_local_environment(start_particle, index)
                local_feature_classifier.compute_atom_feature(start_particle, index)

            if found_new_solution:
                best_particle = copy.deepcopy(start_particle.get_as_dictionary(fields))
                best_particle['energies'][ordering_energy_key] = copy.deepcopy(start_energy)
                best_particle['ads'] = copy.deepcopy(start_particle.get_indices_of_adsorbates())
                best_particles.append(best_particle)
                found_new_solution = False

    accepted_energies.append((accepted_energies[-1][0], total_steps))
    return [best_particle, accepted_energies, best_particles]
