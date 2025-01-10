from typing import List


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

STRATEGIES = {
    "decreasing": _decreasing, 
    "decreasing_v2": _decreasing_v2
}