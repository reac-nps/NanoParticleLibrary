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

def run_monte_carlo(beta, max_steps, start_particle, energy_calculator, local_feature_classifier, adsorbate=None):
    energy_key, local_env_calculator, exchange_operator = setup_monte_carlo(start_particle, energy_calculator,
                                                                            local_feature_classifier)

    if adsorbate is not None: # dictionary with A-Ads and B-Ads bond energy and number of adsorbates
        from Core import Adsorption as ADS
        

        ads = ADS.PlaceAddAtoms(start_particle.get_all_symbols())
        ads.bind_particle(start_particle)
        adsorbate['site_list'] = ads.sites_list
        print('{} Adsorption Sites Identified'.format(len(ads.sites_list)))
        print('{} Energy Gain for {}-Abs Bond'.format(adsorbate[start_particle.get_all_symbols()[0]], start_particle.get_all_symbols()[0]))
        print('{} Energy Gain for {}-Abs Bond'.format(adsorbate[start_particle.get_all_symbols()[1]], start_particle.get_all_symbols()[1]))
        
        index_to_pick = np.arange(len(ads.sites_list))
        random_site_index = np.random.choice(index_to_pick)
        index_to_pick = np.delete(index_to_pick, np.where(index_to_pick == random_site_index)[0])

        initial_adsorbed_sites = [random_site_index]
        site_indices = set(x for x in ads.sites_list[random_site_index])
        while len(initial_adsorbed_sites) < adsorbate['number_of_ads'] and len(index_to_pick) > 0 :
            random_site_index = np.random.choice(index_to_pick)
            site_picked = set(ads.sites_list[random_site_index])
            if len(site_indices.intersection(site_picked)) < adsorbate['shared_atoms']:
                initial_adsorbed_sites.append(random_site_index)
                site_indices = site_indices.union(site_picked)
            index_to_pick = np.delete(index_to_pick, np.where(index_to_pick == random_site_index)[0])
                
        
        #initial_adsorbed_sites = np.random.choice(np.arange(len(adsorbate['site_list'])), adsorbate['number_of_ads'], replace=False)

        def energy_coorection(energy, sites_occupied):
            for site_index in sites_occupied:
                for bond in list(adsorbate['site_list'][site_index]):
                    energy += adsorbate[start_particle.get_symbol(bond)]
            return energy

        


    start_energy = start_particle.get_energy(energy_key)
    #start_energy = energy_coorection(start_energy, initial_adsorbed_sites)
    lowest_energy = start_energy
    #stablest_adsorption_sites = initial_adsorbed_sites
    accepted_energies = [(lowest_energy, 0)] #initial_adsorbed_sites

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

        exchanges = exchange_operator.random_exchange(start_particle)

        start_particle, neighborhood = update_atomic_features(exchanges, local_env_calculator, local_feature_classifier,
                                                              start_particle)
        #new_adsorbed_sites = initial_adsorbed_sites
        accepted_particle = copy.deepcopy(start_particle)
        energy_calculator.compute_energy(start_particle)
        new_energy = start_particle.get_energy(energy_key)
        #new_energy = energy_coorection(new_energy, new_adsorbed_sites)


        delta_e = new_energy - start_energy

        acceptance_rate = min(1, np.exp(-beta * delta_e))
        if np.random.random() < acceptance_rate:
            if found_new_solution:
                if new_energy > start_energy:
                    start_particle.swap_symbols(exchanges)
                    best_particle = copy.deepcopy(start_particle.get_as_dictionary(fields))
                    best_particle['energies'][energy_key] = copy.deepcopy(start_energy)
                    #best_particle['adsorption_sites'] = copy.deepcopy(initial_adsorbed_sites)
                    start_particle.swap_symbols(exchanges)

            start_energy = new_energy
            #initial_adsorbed_sites = new_adsorbed_sites
            accepted_energies.append((new_energy, total_steps))#, new_adsorbed_sites

            if new_energy < lowest_energy:
                no_improvement = 0
                #stablest_adsorption_sites = new_adsorbed_sites
                lowest_energy = new_energy
                found_new_solution = True
            else:
                no_improvement += 1
                found_new_solution = False

        else:
            no_improvement += 1

            # roll back exchanges and make sure features and environments are up-to-date
            start_particle.swap_symbols(exchanges)
            #start_particle = adsorbate_plancer.place_add_atoms(start_particle, atom_to_place, initial_adsorbed_sites)
            start_particle.set_energy(energy_key, start_energy)
            for index in neighborhood:
                local_env_calculator.compute_local_environment(start_particle, index)
                local_feature_classifier.compute_atom_feature(start_particle, index)

            if found_new_solution:
                best_particle = copy.deepcopy(start_particle.get_as_dictionary(fields))
                best_particle['energies'][energy_key] = copy.deepcopy(start_energy)
                #best_particle['adsorption_sites'] = copy.deepcopy(initial_adsorbed_sites)
                found_new_solution = False

    accepted_energies.append((accepted_energies[-1][0], total_steps))#, stablest_adsorption_sites
    return [best_particle, accepted_energies]

def run_monte_carlo_for_adsorbates(beta, max_steps, start_particle, energy_calculator, local_feature_classifier, adsorbate):
    from Core import Adsorption as ADS
    
    energy_key, local_env_calculator, exchange_operator = setup_monte_carlo(start_particle, energy_calculator, local_feature_classifier)

    ads = ADS.PlaceAddAtoms(start_particle.get_all_symbols())
    ads.bind_particle(start_particle)
    adsorbate['site_list'] = ads.sites_list
    number_of_ads = adsorbate['number_of_ads']
    shared_atoms_by_sites = adsorbate['shared_atoms_by_sites']
    initial_adsorbed_sites = random_adsorbate_migration(ads.sites_list, number_of_ads, shared_atoms_by_sites)

    particle_energy = start_particle.get_energy(energy_key)

    start_energy = energy_coorection(particle_energy, start_particle, initial_adsorbed_sites, adsorbate)
    lowest_energy = start_energy
    stablest_adsorption_sites = initial_adsorbed_sites
    accepted_energies = [(lowest_energy, 0, initial_adsorbed_sites)]

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

        new_adsorbed_sites = random_adsorbate_migration(ads.sites_list, number_of_ads, shared_atoms_by_sites)
       
        accepted_particle = copy.deepcopy(start_particle)
        energy_calculator.compute_energy(start_particle)
        new_energy = energy_coorection(particle_energy, start_particle, new_adsorbed_sites, adsorbate)


        delta_e = new_energy - start_energy

        acceptance_rate = min(1, np.exp(-beta * delta_e))
        if np.random.random() < acceptance_rate:
            if found_new_solution:
                if new_energy > start_energy:
                    best_particle = copy.deepcopy(start_particle.get_as_dictionary(fields))
                    best_particle['energies'][energy_key] = copy.deepcopy(start_energy)
                    best_particle['adsorption_sites'] = copy.deepcopy(initial_adsorbed_sites)

            start_energy = new_energy
            initial_adsorbed_sites = new_adsorbed_sites
            accepted_energies.append((new_energy, total_steps, new_adsorbed_sites))

            if new_energy < lowest_energy:
                no_improvement = 0
                stablest_adsorption_sites = new_adsorbed_sites
                lowest_energy = new_energy
                found_new_solution = True
            else:
                no_improvement += 1
                found_new_solution = False

        else:
            no_improvement += 1

            # roll back exchanges and make sure features and environments are up-to-date
            start_energy = energy_coorection(particle_energy,start_particle, initial_adsorbed_sites, adsorbate)
            start_particle.set_energy(energy_key, start_energy)

            if found_new_solution:
                best_particle = copy.deepcopy(start_particle.get_as_dictionary(fields))
                best_particle['energies'][energy_key] = copy.deepcopy(start_energy)
                best_particle['adsorption_sites'] = copy.deepcopy(initial_adsorbed_sites)
                found_new_solution = False

    accepted_energies.append((accepted_energies[-1][0], total_steps, stablest_adsorption_sites))
    return [best_particle, accepted_energies]


def energy_coorection(energy, start_particle, sites_occupied, adsorbate):
    correction = 0
    for site_index in sites_occupied:
        for bond in list(adsorbate['site_list'][site_index]):
            correction += adsorbate[start_particle.get_symbol(bond)]
    return energy + correction

def random_adsorbate_migration(ads_sites_list, number_of_ads, shared_atoms_by_sites):
        adsorbate_sites_indices = np.arange(len(ads_sites_list))

        initial_sites_indices = []
        initial_sites_atoms = set()

        while len(initial_sites_indices) < number_of_ads and len(adsorbate_sites_indices) > 0 :
            random_site_index = np.random.choice(adsorbate_sites_indices)
            random_site_atoms = set(ads_sites_list[random_site_index])
            if len(initial_sites_atoms.intersection(random_site_atoms)) < shared_atoms_by_sites:
                initial_sites_indices.append(random_site_index)
                initial_sites_atoms = initial_sites_atoms.union(random_site_atoms)
            adsorbate_sites_indices = np.delete(adsorbate_sites_indices, np.where(adsorbate_sites_indices == random_site_index)[0])
        return initial_sites_indices