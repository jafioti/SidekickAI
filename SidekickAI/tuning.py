import random, math, inspect, sys, os
from torch.utils.tensorboard import SummaryWriter
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter tuner that uses a genetic algorithim to evolve the hyperparameters
# Hyperparameters to be adjusted are to be passed in as a range (floats are chosen as floats, while ints are chosen as ints)
def evolution_tune(model_class, train_function, hyperparameters, other_parameters, population_size, generations, explore_outside_bounds=True, print_training_progress=False):
    print("Tuning Hyperparameters Through Evolution...")
    # Create initial population
    population = []
    keys = list(hyperparameters.keys())
    ranges = {}
    for genome in range(population_size):
        population.append({})
        for hyperparameter in range(len(keys)):
            assert (isinstance(hyperparameters[keys[hyperparameter]], list) or isinstance(hyperparameters[keys[hyperparameter]], int) or isinstance(hyperparameters[keys[hyperparameter]], float))
            if isinstance(hyperparameters[keys[hyperparameter]], list):
                population[-1][keys[hyperparameter]] = random.randint(hyperparameters[keys[hyperparameter]][0], hyperparameters[keys[hyperparameter]][1]) if (isinstance(hyperparameters[keys[hyperparameter]][0], int) and isinstance(hyperparameters[keys[hyperparameter]][1], int)) else random.uniform(hyperparameters[keys[hyperparameter]][0], hyperparameters[keys[hyperparameter]][1])
                ranges[keys[hyperparameter]] = abs(hyperparameters[keys[hyperparameter]][1] - hyperparameters[keys[hyperparameter]][0])
            else:
                population[-1][keys[hyperparameter]] = hyperparameters[keys[hyperparameter]]
    
    def filter_args_dict_to_function(args_dict, function): # Helper function to filter argument dictionary to only contain arguments that a function accepts
        return {k: v for k, v in args_dict.items() if k in [parameter.name for parameter in inspect.signature(function).parameters.values()]}

    # Run evolution loop
    for generation in range(1, generations + 1):
        print("Generation " + str(generation))
        # Get fitnesses
        fitnesses = []
        for genome in range(len(population)):
            # Create model
            if not isinstance(model_class, str): # Create model using applicable arguments in both the hyperparameters and the other parameters
                model = model_class(**filter_args_dict_to_function(population[genome], model_class), **filter_args_dict_to_function(other_parameters, model_class)).to(device)
            else: # Load model from a path
                model = torch.load(model_class)
            # Run train function
            if not print_training_progress: sys.stdout = open(os.devnull, "w")
            fitnesses.append(train_function(model=model, **filter_args_dict_to_function(population[genome], train_function), **filter_args_dict_to_function(other_parameters, train_function)))
            sys.stdout = sys.__stdout__
            print("Gen " + str(generation) + " Genome " + str(genome + 1) + ": " + str(fitnesses[-1]))
        print("Gen " + str(generation) + " Top Fitness: " + str(max(fitnesses)))

        # Take the best genome and create a new population by adjusting parameters up to 1/5 of the original ranges
        top_genome = population[fitnesses.index(max(fitnesses))]
        print("Gen " + str(generation) + " Top Genome: " + str(top_genome))

        population = []
        for genome in range(population_size):
            population.append({})
            for hyperparameter in range(len(keys)):
                if isinstance(hyperparameters[keys[hyperparameter]], list):
                    population[-1][keys[hyperparameter]] = max(0, top_genome[keys[hyperparameter]] + random.uniform(-ranges[keys[hyperparameter]] / 5, ranges[keys[hyperparameter]] / 5)) if isinstance(hyperparameters[keys[hyperparameter]][0], float) else max(1, top_genome[keys[hyperparameter]] + random.randint(-math.ceil(ranges[keys[hyperparameter]] / 5), math.ceil(ranges[keys[hyperparameter]] / 5)))
                else:
                    population[-1][keys[hyperparameter]] = top_genome[keys[hyperparameter]]