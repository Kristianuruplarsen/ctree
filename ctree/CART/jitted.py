
from numba import njit
import numpy as np


@njit
def search_row_jitted(criterion, y, idx, row):
    ''' Search a row for the best split.
    '''
    score = np.inf

    for obs in row:
        new_rule = row < obs
        y_left = y[new_rule]
        y_right = y[~new_rule]
        
        if len(y_left) > 0:
            score_left = criterion(y_left, np.mean(y_left))
        else:
            score_left = 0
        if len(y_right > 0):
            score_right = criterion(y_right, np.mean(y_right))
        else:
            score_right = 0
        new_score = -(score_left + score_right)

        if new_score < score:
            score = new_score
            rule = new_rule
            split_col = idx
            split_value = obs
    return score, rule, split_col, split_value
