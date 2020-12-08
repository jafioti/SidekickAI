import string, re, jiwer
import numpy as np
from collections import Counter

def f1(pred, answers):
    '''Calculate the F1 score between a string and multiple answer strings\n
    Inputs:
        pred (string or list of strings): The predicted strings
        answers (list of strings or list of lists of strings): The target answer strings
    Outputs:
        score (float): The average F1 score'''
    def score(g_tokens, a_tokens):
        if len(g_tokens) == 0 and len(a_tokens) == 0: return 1
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    # If single instance was passed, make it a batch of one item
    if isinstance(pred, str): pred, answers = [pred], [answers]
    assert len(pred) == len(answers), "There must be an equal amount of predictions and truths"

    scores = []
    for i in range(len(pred)):
        if pred[i] is None or answers[i] is None:
            scores.append(0)
            continue
        scores.append(max([score(normalize_string(pred[i]).split(), normalize_string(a).split()) for a in answers[i]]))
    
    return 100. * sum(scores) / len(scores)

def exact_match(pred, answers):
    '''Calculate the exact match score between a string and multiple answer strings\n
    Inputs:
        pred (string or list of strings): The predicted strings
        answers (list of strings or list of lists of strings): The target answer strings
    Outputs:
        score (float): The average exact match score'''
    def score(pred, answers):
        if pred is None or answers is None:
            return False
        pred = normalize_string(pred)
        for a in answers:
            if pred == normalize_string(a):
                return True
        return False
    
    # If single instance was passed, make it a batch of one item
    if isinstance(pred, str): pred, answers = [pred], [answers]
    assert len(pred) == len(answers), "There must be an equal amount of predictions and truths"

    scores = []
    for i in range(len(pred)):
        scores.append(score(pred[i], answers[i]))
    return 100. * sum(scores) / len(scores)

def word_error_rate(candidate, target, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares target text and
    candidate text in word-level. 
    Inputs:
        candidate (string or list of strings): The candidate(s) text being tested
        target (string or list of strings): The target (correct) string(s)
        ignore_case [Default: False] (bool): Should the metric be case sensitive
        delimeter [Default: ' '] (string): The character signifying a space
    Returns:
        word_error_rate (float): The word error rate between the candidate and the target strings
    WER is defined as:
        WER = (Sw + Dw + Iw) / Nw
    where
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the target
    """
    if ignore_case:
        target = target.lower() if isinstance(target, str) else [trg.lower() for trg in target]
        candidate = candidate.lower() if isinstance(candidate, str) else [cnd.lower() for cnd in candidate]

    return jiwer.wer(target, candidate)


def match_error_rate(candidate, target, ignore_case=False, delimiter=' '):
    """Calculate match error rate (MER). MER compares target text and
    candidate text in word-level. 
    Inputs:
        candidate (string or list of strings): The candidate(s) text being tested
        target (string or list of strings): The target (correct) string(s)
        ignore_case [Default: False] (bool): Should the metric be case sensitive
        delimeter [Default: ' '] (string): The character signifying a space
    Returns:
        match_error_rate (float): The match error rate between the candidate and the target strings
    """
    if ignore_case:
        target = target.lower() if isinstance(target, str) else [trg.lower() for trg in target]
        candidate = candidate.lower() if isinstance(candidate, str) else [cnd.lower() for cnd in candidate]

    return jiwer.mer(target, candidate)

def word_information_loss(candidate, target, ignore_case=False, delimiter=' '):
    """Calculate word information loss (WIL). WIL compares target text and
    candidate text in word-level. 
    Inputs:
        candidate (string or list of strings): The candidate(s) text being tested
        target (string or list of strings): The target (correct) string(s)
        ignore_case [Default: False] (bool): Should the metric be case sensitive
        delimeter [Default: ' '] (string): The character signifying a space
    Returns:
        word_information_loss (float): The word information loss between the candidate and the target strings
    """
    if ignore_case:
        target = target.lower() if isinstance(target, str) else [trg.lower() for trg in target]
        candidate = candidate.lower() if isinstance(candidate, str) else [cnd.lower() for cnd in candidate]

    return jiwer.wil(target, candidate)

def asr_metrics(candidate, target, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER), match error rate (MER), and word information loss (WIL).
    Inputs:
        candidate (string or list of strings): The candidate(s) text being tested
        target (string or list of strings): The target (correct) string(s)
        ignore_case [Default: False] (bool): Should the metric be case sensitive
        delimeter [Default: ' '] (string): The character signifying a space
    Returns:
        word_error_rate (float): The word error rate between the candidate and the target strings
        match_error_rate (float): The match error rate between the candidate and the target strings
        word_information_loss (float): The word information loss between the candidate and the target strings
    """
    if ignore_case:
        target = target.lower() if isinstance(target, str) else [trg.lower() for trg in target]
        candidate = candidate.lower() if isinstance(candidate, str) else [cnd.lower() for cnd in candidate]

    return jiwer.compute_measures(target, candidate)


# Helper function to normalize text (lowercase, remove punctuation, remove articles, and fix white space)
def normalize_string(s):
    '''Normalize the string (lowercase, remove punctuation, remove articles, and fix white space)\n
    Inputs:
        s (string): A normal string to be normalized
    Outputs:
        s (string): The normalized string'''
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))