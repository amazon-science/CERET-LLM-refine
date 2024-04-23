from collections import Counter
from typing import Dict, List, Optional, Set, Tuple


#############################################################################


def get_freq_and_dist(data_list):
    """return [frequency, distribution]"""
    frequency = Counter(data_list)
    total_elements = len(data_list)
    distribution = {element: (count / total_elements) * 100 for element, count in frequency.items()}
    return frequency, distribution


def get_ls_mean(ls: List[float]):
    return sum(ls) / len(ls)
