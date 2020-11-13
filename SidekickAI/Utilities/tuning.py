import random, math, sys, os, copy
from torch.utils.tensorboard import SummaryWriter
import torch

from SidekickAI.Utilities.utils import filter_args_dict_to_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyperparameter tuner that uses a genetic algorithim to evolve the hyperparameters
# Hyperparameters to be adjusted are to be passed in as a range (floats are chosen as floats, while ints are chosen as ints)
def evolution_tune(model_class, train_function, hyperparameters, other_parameters, population_size, generations, explore_outside_bounds=True, print_training_progress=False):
    '''A hyperparameter tuner that uses a genetic algorithim\n
    Inputs:
        model_class (class): The class of the model (not an instance)
        train_function (function): The training function, which handles initializing the optimizer and any lr scheduling as well
        hyperparameters (dict): All of the hyperparameters to be tuned, passed into either the model or the train function. Ones to be adjusted are to be passed in as a range (floats are chosen as floats, while ints are chosen as ints)
        other_parameters (dict): Any other parameters to be passed into either the model or the train function
        population_size (int): The size of the population of hyperparameter samples for each generation
        generations (int): The number of generations to search for
        explore_outside_bounds (bool) [default=True]: Whether to search the parameters outside given bounds in subsequent generations
        print_training_progress (bool) [default=False]: Whether to allow the training function to print'''
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
                model = torch.load(model_class).to(device)
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

# Random search over hyperparameters
def random_tune(model_class, train_function, hyperparameters, other_parameters, num_samples, print_training_progress=False, top_samples_to_print=3):
    '''A hyperparameter tuner that chooses random samples of hyperparameters\n
    Inputs:
        model_class (class): The class of the model (not an instance)
        train_function (function): The training function, which handles initializing the optimizer and any lr scheduling as well
        hyperparameters (dict): All of the hyperparameters to be tuned, passed into either the model or the train function. Ones to be adjusted are to be passed in as a range (floats are chosen as floats, while ints are chosen as ints)
        other_parameters (dict): Any other parameters to be passed into either the model or the train function
        num_samples (int): The total number of samples to test
        print_training_progress (bool) [default=False]: Whether to allow the training function to print
        top_samples_to_print (int) [default=3]: The number of top samples to print at the end of testing'''
    print("Running Random Search Over Hyperparameters...")
    top_samples_to_print = min(top_samples_to_print, num_samples) # Limit the print samples to the number of samples we are testing
    # Pick samples
    samples = []
    keys = list(hyperparameters.keys())
    for sample in range(num_samples):
        samples.append({})
        for hyperparameter in range(len(keys)):
            if isinstance(hyperparameters[keys[hyperparameter]], list):
                if isinstance(hyperparameters[keys[hyperparameter]][0], int) and isinstance(hyperparameters[keys[hyperparameter]][0], int):
                    samples[-1][keys[hyperparameter]] = random.randint(hyperparameters[keys[hyperparameter]][0], hyperparameters[keys[hyperparameter]][1])
                else:
                    samples[-1][keys[hyperparameter]] = random.uniform(hyperparameters[keys[hyperparameter]][0], hyperparameters[keys[hyperparameter]][1])
            else:
                samples[-1][keys[hyperparameter]] = hyperparameters[keys[hyperparameter]]
    
    # Run samples through training
    fitnesses = []
    for sample in range(len(samples)):
        model = model_class

        # Create model
        if not isinstance(model_class, str): # Create model using applicable arguments in both the hyperparameters and the other parameters
            model = model_class(**filter_args_dict_to_function(samples[sample], model_class), **filter_args_dict_to_function(other_parameters, model_class)).to(device)
        else: # Load model from a path
            model = torch.load(model_class).to(device)

        # Run training
        if not print_training_progress: sys.stdout = open(os.devnull, "w")
        fitnesses.append(train_function(model=model, **filter_args_dict_to_function(samples[sample], train_function), **filter_args_dict_to_function(other_parameters, train_function)))
        sys.stdout = sys.__stdout__

        print("Sample " + str(sample) + ": " + str(fitnesses[-1]) + ", " + str(samples[sample]))

    if isinstance(top_samples_to_print, int) and top_samples_to_print > 0:
        print("TOP " + str(top_samples_to_print) + " HYPERPARAMETER SAMPLES:")
        # Sort by top fitnesses
        zipped_lists = list(zip(fitnesses, samples))
        zipped_lists.sort(reverse=True, key=lambda x: x[0])
        fitnesses, samples = zip(*zipped_lists)
        for i in range(top_samples_to_print):
            print(str(i + 1) + " Fitness: " + str(fitnesses[i]) + ", " + str(samples[i]))

# Grid search over hyperparameters
def grid_tune(model_class, train_function, hyperparameters, other_parameters, print_training_progress=False, top_samples_to_print=None):
    '''A hyperparameter tuner that runs grid search on all possible hyperparameter combinations\n
    Inputs:
        model_class (class): The class of the model (not an instance)
        train_function (function): The training function, which handles initializing the optimizer and any lr scheduling as well
        hyperparameters (dict): All of the hyperparameters to be tuned, passed into either the model or the train function. Ones to be adjusted are to be passed in as a list of discrete values
        other_parameters (dict): Any other parameters to be passed into either the model or the train function
        print_training_progress (bool) [default=False]: Whether to allow the training function to print
        top_samples_to_print (int) [default=3]: The number of top samples to print at the end of testing'''
    print("Running Grid Search Over Hyperparameters...")
    # Pick samples
    samples = [copy.deepcopy(hyperparameters)]
    keys = list(hyperparameters.keys())
    for hyperparameter in range(len(keys)):
        # If this is a list of steps, make a new configuration if we have not already added it
        if isinstance(hyperparameters[keys[hyperparameter]], list):
            for config in range(len(hyperparameters[keys[hyperparameter]])):
                current_config = copy.deepcopy(hyperparameters)
                current_config[keys[hyperparameter]] = hyperparameters[keys[hyperparameter]][config]
                # Check if we already added this config
                if current_config not in samples: samples.append(current_config)
                
    top_samples_to_print = min(top_samples_to_print, len(samples)) # Limit the print samples to the number of samples we are testing

    # Run samples through training
    fitnesses = []
    for sample in range(len(samples)):
        model = model_class

        # Create model
        if not isinstance(model_class, str): # Create model using applicable arguments in both the hyperparameters and the other parameters
            model = model_class(**filter_args_dict_to_function(samples[sample], model_class), **filter_args_dict_to_function(other_parameters, model_class)).to(device)
        else: # Load model from a path
            model = torch.load(model_class).to(device)

        # Run training
        if not print_training_progress: sys.stdout = open(os.devnull, "w")
        fitnesses.append(train_function(model=model, **filter_args_dict_to_function(samples[sample], train_function), **filter_args_dict_to_function(other_parameters, train_function)))
        sys.stdout = sys.__stdout__

        print("Sample " + str(sample) + ": " + str(fitnesses[-1]) + ", " + str(samples[sample]))

    if isinstance(top_samples_to_print, int) and top_samples_to_print > 0:
        print("TOP " + str(top_samples_to_print) + " HYPERPARAMETER SAMPLES:")
        # Sort by top fitnesses
        zipped_lists = list(zip(fitnesses, samples))
        zipped_lists.sort(reverse=True, key=lambda x: x[0])
        fitnesses, samples = zip(*zipped_lists)
        for i in range(top_samples_to_print):
            print(str(i + 1) + " Fitness: " + str(fitnesses[i]) + ", " + str(samples[i]))