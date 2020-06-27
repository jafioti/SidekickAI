
# Contains functions for NLP evaluation

# Takes in output (a list of tokens or indexes) and a target (also a list of tokens or indexes)
def f1_score(output, target):
    if len(output) > 0 and len(target) > 0:
        # Get precision and recall
        precision = sum([1 if output[i] in target else 0 for i in range(len(output))]) / len(output)
        recall = sum([1 if target[i] in output else 0 for i in range(len(target))]) / len(target)
        # Combine together to get f1 score
        f1 = (2 * precision * recall) / (precision + recall) if (precision > 0 and recall > 0) else 0
        return f1
    else:
        return 0