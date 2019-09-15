
from numba import njit
import numpy as np


def search_row_nojit(criterion, y, row, min_leaf_size):
        score = np.inf
        leaf_candidate = True

        for obs in row:
            new_rule = row < obs
            y_left = y[new_rule]
            y_right = y[~new_rule]

            if len(y_left) < min_leaf_size \
            or len(y_right) < min_leaf_size:
                continue
            
            score_left = criterion(y_left, np.mean(y_left))
            score_right = criterion(y_right, np.mean(y_right))
            new_score = score_left + score_right
    
            if new_score < score:
                score = new_score
                rule = new_rule
                split_value = obs
                leaf_candidate = False

        if leaf_candidate:
            # If there were no good splits
            # this is a leaf candidate
            score = None
            rule = None
            split_value = None

        return score, rule, split_value, leaf_candidate


search_row_jit = njit(search_row_nojit)

