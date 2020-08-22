import string, re
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
        candidate (string): The candidate text being tested
        target (string): The target (correct) string
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
    if ignore_case == True:
        target = target.lower()
        candidate = candidate.lower()

    ref_words = target.split(delimiter)
    hyp_words = candidate.split(delimiter)

    if len(ref_words) == 0:
        raise ValueError("target's word number should be greater than 0.")

    edit_distance = levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance) / len(ref_words)


def character_error_rate(candidate, target, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares target text and
    candidate text in char-level. \n
    Inputs:
        candidate (string): The candidate text being tested
        target (string): The target (correct) string
        ignore_case [Default: False] (bool): Should the metric be case sensitive
        remove_space [Default: False] (bool): Should spaces be removed before the metric is calculated
    Returns:
        character_error_rate (float): The character error rate between the candidate and the target strings
    CER is defined as:
        CER = (Sc + Dc + Ic) / Nc
    where
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the target
    """
    if ignore_case == True:
        target = target.lower()
        candidate = candidate.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    target = join_char.join(filter(None, target.split(' ')))
    candidate = join_char.join(filter(None, candidate.split(' ')))

    if len(target) == 0:
        raise ValueError("Length of target should be greater than 0.")

    edit_distance = levenshtein_distance(target, candidate)
    return float(edit_distance) / len(target)

def levenshtein_distance(sentence1, sentence2):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other.\n
    Inputs:
        sentence1 (string): The first sentence
        sentence2 (string): The second sentence
    Returns:
        levenshtein_distance (float): The levenshtein distance between the two strings
    """
    import numpy as np
    s1_len = len(sentence1)
    s2_len = len(sentence2)

    # special case
    if sentence1 == sentence2:
        return 0
    if s1_len == 0:
        return s2_len
    if s2_len == 0:
        return s1_len

    if s1_len < s2_len: # Ensure that sentence 1 is longer or equal
        sentence1, sentence2 = sentence2, sentence1
        s1_len, s2_len = s2_len, s1_len

    # use O(min(s1_len, s2_len)) space
    distance = np.zeros((2, s2_len + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,s2_len + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, s1_len + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, s2_len + 1):
            if sentence1[i - 1] == sentence2[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[s1_len % 2][s2_len]


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