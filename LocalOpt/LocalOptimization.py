from Core.LocalEnvironmentCalculator import NeighborCountingEnvironmentCalculator
from LocalOpt.GuidedExchangeOperator import GuidedExchangeOperator
from Core.LocalEnvironmentFeatureClassifier import TopologicalEnvironmentClassifier


def setup_local_optimization(start_particle, energy_calculator, environment_energies, local_feature_classifier=None):
    symbols = start_particle.get_all_symbols()
    local_env_calculator = NeighborCountingEnvironmentCalculator(symbols)
    if local_feature_classifier is None:
        local_feature_classifier = TopologicalEnvironmentClassifier(local_env_calculator, symbols)

    local_env_calculator.compute_local_environments(start_particle)
    local_feature_classifier.compute_feature_vector(start_particle)

    feature_key = local_feature_classifier.get_feature_key()
    energy_calculator.compute_energy(start_particle)

    exchange_operator = GuidedExchangeOperator(environment_energies, feature_key)
    exchange_operator.bind_particle(start_particle)

    energy_key = energy_calculator.get_energy_key()

    return energy_key, local_env_calculator, local_feature_classifier, exchange_operator


def update_atomic_features(index1, index2, local_env_calculator, local_feature_classifier, particle):
    neighborhood = {index1, index2}
    neighborhood = neighborhood.union(particle.neighbor_list[index1])
    neighborhood = neighborhood.union(particle.neighbor_list[index2])

    for index in neighborhood:
        local_env_calculator.compute_local_environment(particle, index)
        local_feature_classifier.compute_atom_feature(particle, index)

    local_feature_classifier.compute_feature_vector(particle, recompute_atom_features=False)
    return particle, neighborhood


def local_optimization(start_particle, energy_calculator, environment_energies, local_feature_classifier=None):
    energy_key, local_env_calculator, local_feature_classifier, exchange_operator = setup_local_optimization(
        start_particle, energy_calculator, environment_energies, local_feature_classifier)

    start_energy = start_particle.get_energy(energy_key)
    accepted_energies = [(start_energy, 0)]

    step = 0
    while True:
        step += 1
        index1, index2 = exchange_operator.guided_exchange(start_particle)
        exchanged_indices = [index1, index2]

        start_particle, neighborhood = update_atomic_features(index1, index2, local_env_calculator,
                                                              local_feature_classifier, start_particle)
        exchange_operator.update(start_particle, neighborhood, exchanged_indices)

        energy_calculator.compute_energy(start_particle)
        new_energy = start_particle.get_energy(energy_key)

        if new_energy < start_energy:
            start_energy = new_energy
            accepted_energies.append((new_energy, step))
        else:
            accepted_energies.append((start_energy, step))

            # roll back last exchange and make sure features and environments are up-to-date
            start_particle.swap_symbols([(index1, index2)])
            start_particle.set_energy(energy_key, start_energy)
            for index in neighborhood:
                local_env_calculator.compute_local_environment(start_particle, index)
                local_feature_classifier.compute_atom_feature(start_particle, index)

            break

    return [start_particle, accepted_energies]
