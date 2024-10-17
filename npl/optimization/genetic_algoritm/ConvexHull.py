import numpy as np
from GA.CutAndSpliceOperator import CutAndSpliceOperator
from GA.ExchangeOperator import ExchangeOperator
from GA.MutationOperator import MutationOperator

import copy
import pickle


def locate_convex_hull(start_population, unsuccessful_gens_for_convergence,
                       energy_calculator, local_env_calculator, local_feature_classifier):
    gens_since_last_success = 0
    symbols = ['Pt', 'Au']

    mutation_operator = MutationOperator(0.5, symbols)
    exchange_operator = ExchangeOperator(0.5)
    cut_and_splice_operator = CutAndSpliceOperator(0, 10)

    population = copy.deepcopy(start_population)
    for p in population.population.values():
        local_env_calculator.compute_local_environments(p)
        local_feature_classifier.compute_feature_vector(p)
        energy_calculator.compute_energy(p)

    energy_log = [population.get_convex_hull()]

    cur_generation = 0
    while gens_since_last_success < unsuccessful_gens_for_convergence:
        cur_generation += 1

        if cur_generation % 1000 == 0:
            energy_log.append(population.get_convex_hull())
            pickle.dump(energy_log, open("energy_log.pkl", 'wb'))

        p = np.random.random()
        if p < 0.6:
            # choose two parents for cut and splice
            parent1, parent2 = population.tournament_selection(2, 5)
            new_particle = cut_and_splice_operator.cut_and_splice(parent1, parent2, False)

        elif p < 0.8:
            # random exchange
            parent = population.random_selection(1)[0]
            new_particle = exchange_operator.random_exchange(parent)

        else:
            # random mutation
            parent = population.gaussian_tournament(1, 5)[0]
            new_particle = mutation_operator.random_mutation(parent)

        # check that it is not a pure particle
        if new_particle.is_pure():
            continue

        # TODO insert local optimization
        local_env_calculator.compute_local_environments(new_particle)
        local_feature_classifier.compute_feature_vector(new_particle)
        energy_calculator.compute_energy(new_particle)

        successful_offspring = False
        niche = population.get_niche(new_particle)

        if new_particle.get_energy('BRR') < population[niche].get_energy('BRR'):
            print("success in generation: {}".format(cur_generation))
            print("New Energy: {}".format(new_particle.get_energy('BRR')))
            successful_offspring = True
            population.add_offspring(new_particle)

        # reset counters and log energy
        if successful_offspring:
            gens_since_last_success = 0
        else:
            gens_since_last_success += 1

    return [population, energy_log, cur_generation]
