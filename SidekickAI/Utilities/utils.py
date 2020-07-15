# A bunch of  random useful functions with no good place in the library

def load_model(path, model_class):
    '''Loads a model from the model class, and the checkpoint file, which inclues the state dict and the hyperparameters'''
    import os
    from torch import load
    assert os.path.isfile(path), "The checkpoint file does not exist at the selected path"
    checkpoint = load(path)
    assert "state_dict" in list(checkpoint.keys()), "The checkpoint does not contain a state_dict"
    assert "hyperparameters" in list(checkpoint.keys()), "The checkpoint does not contain hyperparameters"
    model = model_class(**filter_args_dict_to_function(checkpoint["hyperparameters"], model_class))
    model.load_state_dict(checkpoint["state_dict"])
    return model

def save_model(path, model):
    from torch import save
    '''Saves a model to a file containing the state dict and the hyperparameters'''
    assert hasattr(model, "state_dict"), "The model does not have a state dict"
    assert hasattr(model, "hyperparameters"), "The model does not have a 'hyperparameters' variable"
    save({"state_dict":model.state_dict(), "hyperparameters":model.hyperparameters}, path)

def count_parameters(model, trainable_only=True, format_as_string=True):
    '''Counts the parameters in a model'''
    params = sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
    if format_as_string:
        #  Cleanly format as string
        for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
            if abs(params) < 1000.0:
                return "%3.1f%s" % (params, unit)
            params /= 1000.0
        return "%.1f%s" % (params, 'Yi')
    return params

def get_learning_rate(optimizer, round_digits=4):
    '''Returns the average learning rate for an optimizer'''
    lrs = [group['lr'] for group in optimizer.param_groups]
    return round(sum(lrs) / len(lrs), round_digits)

def filter_args_dict_to_function(args_dict, function): 
    '''Filters argument dictionary to only contain arguments that a function accepts'''
    import inspect
    return {k: v for k, v in args_dict.items() if k in [parameter.name for parameter in inspect.signature(function).parameters.values()]}

def lr_decay(optimizer, lr_decay):
    ''' Decay an optimizer's learning rates by the lr_decay factor. \n
        new_lr = old_lr * lr_decay'''
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer

class ExponentialAverage(object):
    """Keep exponential weighted averages."""
    def __init__(self, beta=0.99):
        self.beta = beta
        self.moment = 0
        self.value = 0
        self.t = 0

    def state_dict(self):
        return vars(self)

    def load(self, state_dict):
        for k, v in state_dict.items():
            self.__setattr__(k, v)

    def update(self, val):
        self.t += 1
        self.moment = self.beta * self.moment + (1 - self.beta) * val
        # bias correction
        self.value = self.moment / (1 - self.beta ** self.t)

    def reset(self):
        self.moment = 0
        self.value = 0
        self.t = 0