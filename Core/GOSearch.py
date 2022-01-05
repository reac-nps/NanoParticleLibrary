from Core import LocalEnvironmentCalculator as LEC
from Core import LocalEnvironmentFeatureClassifier as LFC
from Core import EnergyCalculator as EC
from Core import GlobalFeatureClassifier as GFC

import time


class GOSearch:
    def __init__(self, find_global_minimum, create_start_configuration):
        self.find_global_minimum = find_global_minimum
        self.create_start_configuration = create_start_configuration

    def fit_energy_expression(self, training_set, symbols):
        raise NotImplementedError

    def build_args_list_for_gm_search(self, args_gm, args_start):
        raise NotImplementedError

    def build_kwargs_list_for_gm_search(self, additional_kwargs):
        raise NotImplementedError

    def run_multiple_simulations(self, n_runs, args_gm=None, kwargs_gm=None, args_start=None):
        run_times = []
        results = []
        for i in range(n_runs):
            print("Run: {}".format(i))

            args = self.build_args_list_for_gm_search(args_gm, args_start)
            kwargs = self.build_kwargs_list_for_gm_search(kwargs_gm)

            start_time = time.perf_counter()
            result = self.find_global_minimum(*args, **kwargs)
            end_time = time.perf_counter()

            run_time = end_time - start_time
            print("Runtime: {}".format(run_time))

            run_times.append(run_time)
            results.append(result)

        return results, run_times


class MCSearch(GOSearch):
    def __init__(self, find_global_minimum, create_start_configuration, beta, n_steps):
        GOSearch.__init__(self, find_global_minimum, create_start_configuration)
        self.beta = beta
        self.n_steps = n_steps
        self.energy_calculator = None
        self.local_feature_classifier = None

    def fit_energy_expression(self, training_set, symbols):
        local_env_calculator = LEC.NeighborCountingEnvironmentCalculator(symbols)
        global_feature_classifier = GFC.TopologicalFeatureClassifier(symbols)

        self.energy_calculator = EC.BayesianRRCalculator(global_feature_classifier.get_feature_key())
        self.local_feature_classifier = LFC.TopologicalEnvironmentClassifier(local_env_calculator, symbols)

        for p in training_set:
            global_feature_classifier.compute_feature_vector(p)

        self.energy_calculator.fit(training_set, 'EMT')

        n_atoms = sum(list(training_set[0].get_stoichiometry().values()))
        lin_coef = self.energy_calculator.get_coefficients()
        topological_coefficients, _ = EC.compute_coefficients_for_linear_topological_model(lin_coef, symbols, n_atoms)
        print(topological_coefficients)

        self.energy_calculator.set_coefficients(topological_coefficients)
        self.energy_calculator.set_feature_key(self.local_feature_classifier.get_feature_key())

        return

    def build_args_list_for_gm_search(self, additional_args, args_start):
        if args_start is None:
            start_config = self.create_start_configuration()
        else:
            start_config = self.create_start_configuration(*args_start)

        args = [self.beta, self.n_steps, start_config, self.energy_calculator, self.local_feature_classifier]
        if additional_args is not None:
            return args + additional_args
        else:
            return args

    def build_kwargs_list_for_gm_search(self, additional_kwargs):
        if additional_kwargs is not None:
            return additional_kwargs
        else:
            return dict()

    def set_beta(self, beta):
        self.beta = beta

    def set_n_steps(self, n_steps):
        self.n_steps = n_steps


class GASearch(GOSearch):
    def __init__(self, genetic_algorithm, create_start_population, unsuccessful_gens_for_convergence):
        GOSearch.__init__(self, genetic_algorithm, create_start_population)
        self.unsuccessful_steps_for_convergence = unsuccessful_gens_for_convergence

        self.local_env_calculator = None
        self.energy_calculator = None
        self.local_feature_classifier = None
        self.total_energies = None

    def fit_energy_expression(self, training_set, symbols):
        local_env_calculator = LEC.NeighborCountingEnvironmentCalculator(symbols)
        global_feature_classifier = GFC.TopologicalFeatureClassifier(symbols)

        self.energy_calculator = EC.BayesianRRCalculator(global_feature_classifier.get_feature_key())
        self.local_feature_classifier = LFC.TopologicalEnvironmentClassifier(local_env_calculator, symbols)
        self.local_env_calculator = local_env_calculator

        for p in training_set:
            global_feature_classifier.compute_feature_vector(p)

        self.energy_calculator.fit(training_set, 'EMT')

        n_atoms = sum(list(training_set[0].get_stoichiometry().values()))
        lin_coef = self.energy_calculator.get_coefficients()
        print(lin_coef)
        topological_coefficients, self.total_energies = EC.compute_coefficients_for_linear_topological_model(
            lin_coef, symbols, n_atoms)

        self.energy_calculator.set_coefficients(topological_coefficients)
        self.energy_calculator.set_feature_key(self.local_feature_classifier.get_feature_key())

        return

    def build_args_list_for_gm_search(self, additional_args, args_start):
        if args_start is None:
            start_population = self.create_start_configuration()
        else:
            start_population = self.create_start_configuration(*args_start)
        args = [start_population, self.unsuccessful_steps_for_convergence, self.energy_calculator,
                self.local_env_calculator, self.local_feature_classifier, self.total_energies]

        if additional_args is not None:
            return args + additional_args
        else:
            return args

    def build_kwargs_list_for_gm_search(self, additional_kwargs):
        if additional_kwargs is not None:
            return additional_kwargs
        else:
            return dict()


class GuidedSearch(GOSearch):
    def __init__(self, find_global_minimum, create_start_configuration):
        GOSearch.__init__(self, find_global_minimum, create_start_configuration)
        self.total_energies = None
        self.energy_calculator = None

    def fit_energy_expression(self, training_set, symbols, energy_key='EMT'):
        global_feature_classifier = GFC.TopologicalFeatureClassifier(symbols)

        self.energy_calculator = EC.BayesianRRCalculator(global_feature_classifier.get_feature_key())

        for p in training_set:
            global_feature_classifier.compute_feature_vector(p)

        self.energy_calculator.fit(training_set, energy_key)

        n_atoms = sum(list(training_set[0].get_stoichiometry().values()))
        lin_coef = self.energy_calculator.get_coefficients()
        print(lin_coef)
        topological_coefficients, self.total_energies = EC.compute_coefficients_for_linear_topological_model(
            lin_coef, symbols, n_atoms)

        self.energy_calculator.set_coefficients(topological_coefficients)
        self.energy_calculator.set_feature_key('TEC')

        return

    def build_args_list_for_gm_search(self, additional_args, args_start):
        if args_start is None:
            start_config = self.create_start_configuration()
        else:
            start_config = self.create_start_configuration(*args_start)

        args = [start_config, self.energy_calculator, self.total_energies]
        if additional_args is not None:
            return args + additional_args
        else:
            return args

    def build_kwargs_list_for_gm_search(self, additional_kwargs):
        if additional_kwargs is not None:
            return additional_kwargs
        else:
            return dict()