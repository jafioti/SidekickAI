
def count_parameters(model, trainable_only=True): # Counts the parameters in the model
    params = sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
    #  Cleanly format
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(params) < 1000.0:
            return "%3.1f%s" % (params, unit)
        params /= 1000.0
    return "%.1f%s" % (params, 'Yi')

def get_learning_rate(optimizer, round_digits=4): # Returns the average learning rate for an optimizer
    lrs = [group['lr'] for group in optimizer.param_groups]
    return round(sum(lrs) / len(lrs), round_digits)