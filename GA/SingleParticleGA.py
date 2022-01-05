import numpy as np
from GA.CutAndSpliceOperator import CutAndSpliceOperator
from GA.ExchangeOperator import ExchangeOperator
from LocalOpt.LocalOptimization import local_optimization


def compute_fitness(particle, min_energy, max_energy, energy_key):
    if max_energy == min_energy:
        return 0
    normalized_energy = (particle.get_energy(energy_key) - min_energy) / (max_energy - min_energy)
    return np.exp(-3 * normalized_energy)


def run_single_particle_ga(start_population, unsuccessful_gens_for_convergence, energy_calculator, local_env_calculator,
                           local_feature_classifier, environment_energies):
    unsuccessful_gens = 0
    energy_key = energy_calculator.get_energy_key()

    for p in start_population:
        local_env_calculator.compute_local_environments(p)
        local_feature_classifier.compute_feature_vector(p)
        energy_calculator.compute_energy(p)

    exchange_operator = ExchangeOperator(0.5)
    cut_and_splice_operator = CutAndSpliceOperator(10)

    cur_population = start_population
    generation = 0
    best_energies = []
    energy_evaluations = len(start_population)

    cur_population.sort(key=lambda x: x.get_energy(energy_key))
    best_energies.append((cur_population[0].get_energy(energy_key), energy_evaluations))
    while unsuccessful_gens < unsuccessful_gens_for_convergence:
        if generation % 1 == 0:
            print("Generation: {}".format(generation))
            print(energy_evaluations)

        min_energy = cur_population[0].get_energy(energy_key)
        max_energy = cur_population[-1].get_energy(energy_key)
        generation += 1

        fitness_values = np.array([compute_fitness(p, min_energy, max_energy, energy_key) for p in cur_population])
        if np.sum(fitness_values) == 0:
            break
        fitness_values /= np.sum(fitness_values)

        while True:
            p = np.random.rand()
            if p < 0.4:
                parent1, parent2 = np.random.choice(cur_population, 2, replace=False, p=fitness_values)
                new_offspring = cut_and_splice_operator.cut_and_splice(parent1, parent2)
            else:
                parent = np.random.choice(cur_population, 1, p=fitness_values)[0]
                new_offspring = exchange_operator.random_exchange(parent)

            new_offspring, energies = local_optimization(new_offspring,
                                                         energy_calculator,
                                                         environment_energies)

            energy_evaluations += energies[-1][1]

            new_energy = new_offspring.get_energy(energy_key)
            unique = True
            for particle in cur_population:
                if np.abs(new_energy - particle.get_energy(energy_key)) < 1e-6:
                    unique = False
                    break
            if unique:
                break

        cur_population.append(new_offspring)

        cur_population.sort(key=lambda x: x.get_energy(energy_key))
        cur_population = cur_population[:-1]

        if cur_population[0].get_energy(energy_key) == best_energies[-1][0]:
            unsuccessful_gens += 1
        else:
            unsuccessful_gens = 0
            best_energies.append((cur_population[0].get_energy(energy_key), energy_evaluations))
            print("New best energy: {}".format(cur_population[0].get_energy(energy_key)))

    return [best_energies, cur_population[0], energy_evaluations]
