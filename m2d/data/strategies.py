import math
from typing import List


def _confidence_to_thres_upper_bound(
    confidence: float, 
    vocab_size: int = 128256
):
    """ Assume uniform over N - 1 options """
    # Input validation
    if not (0 < confidence <= 1):
        raise ValueError("confidence must be in (0, 1]")
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    
    # If confidence is 1.0, entropy is 0 (all probability on one token)
    if confidence == 1.0:
        return 0.0
    
    # Define variables for clarity
    p = confidence  # Probability of the top token
    N = vocab_size  # Total vocabulary size
    
    # Probability of each of the remaining (N - 1) tokens
    p_other = (1 - p) / (N - 1)
    
    # Calculate entropy: H = - [p * ln(p) + (1 - p) * ln(p_other)]
    term1 = p * math.log(p)  # Contribution from the top token
    term2 = (1 - p) * math.log(p_other)  # Contribution from the remaining tokens
    entropy = - (term1 + term2)
    
    return entropy

MIN_THRESHOLD = _confidence_to_thres_upper_bound(confidence=0.99)

def _decreasing(
    entropy: List[float], 
    max_steps: int, 
    ratio_threshold: float = 0.3, 
    **kwargs
):
    steps = []
    last_cnt = 0
    last_max = -1
    for e in entropy:
        if e > ratio_threshold * last_max or last_cnt >= max_steps:
            # start new segment
            steps.append(last_cnt)
            last_cnt = 1
            last_max = e
        else:
            # keep old segment
            last_cnt += 1
    
    if last_cnt > 0:
        steps.append(last_cnt)
    steps = steps[1:]
    assert sum(steps) == len(entropy)
    return steps

def _decreasing_v2(
    entropy: List[float], 
    max_steps: int, 
    min_threshold: float = 0.05, 
    ratio_threshold: float = 0.3, 
    **kwargs
):
    """
    We want a heuristic like the following:
        1. Each segment should not exceed `max_steps`
        2. We want the first token in a segment to be the biggest unless all of them are small
        3. We want to minimize the number of segments if possible
    There are two types of segments:
        1. Low entropy segment: all tokens are low in entropy
            a. All tokens are below a certain threshold
        2. Compressed segment: the entropy of the first token is high
            a. The entropies of tokens after the first are at most 0.x times large
    """
    steps = []
    last_cnt = 0
    last_max = -1
    for e in entropy:
        if (e < min_threshold or e < last_max * ratio_threshold) \
            and last_cnt < max_steps:
            # append to current one
            last_cnt += 1
            last_max = max(last_max, e)
        else:
            # start a new one
            if last_cnt > 0:
                steps.append(last_cnt)
            last_cnt = 1
            last_max = e
            
    if last_cnt > 0:
        steps.append(last_cnt)
    assert sum(steps) == len(entropy)
    return steps

def _small_group(
    entropy: List[float], 
    max_steps: int, 
    min_threshold: float = MIN_THRESHOLD, 
    **kwargs
):
    steps = []
    last_cnt = 0
    for e in entropy:
        if e > min_threshold:
            if last_cnt > 0:
                steps.append(last_cnt)
                last_cnt = 0
            steps.append(1)
        else:
            if last_cnt == max_steps:
                steps.append(last_cnt)
                last_cnt = 0
            last_cnt += 1
    if last_cnt > 0:
        steps.append(last_cnt)
    assert sum(steps) == len(entropy)
    return steps


STRATEGIES = {
    "decreasing": _decreasing, 
    "decreasing_v2": _decreasing_v2, 
    "small_group": _small_group
}