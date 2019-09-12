
import numpy as np

import multiprocessing

from numba import njit
from .jitted import search_row_jitted


def MSE(yhat, y):
    return np.mean((y - yhat)**2)

def MAE(yhat, y):
    return np.mean(np.abs(y - yhat))



class Partition:
    ''' This class wraps the CART Node
        for abstraction.
    '''
    _CRITERIA = {'mse': MSE,
                 'mae': MAE}

    def __init__(self,
                 criterion = 'mse', 
                 jit = True,
                 min_leaf_size = 5
                ):
        self.criterion_str = criterion
        self.jit = jit

        self.criterion = self._CRITERIA[criterion]
        if jit:
            self.criterion = njit(self.criterion)

        self.min_leaf_size = min_leaf_size
        self.dtree = None

    def fit(self, X, y):
        self.dtree = Node(X, y, jit = self.jit, criterion=self.criterion, 
                          min_leaf_size=self.min_leaf_size)
        return self

    def predict(self, X):
        preds = []
        for xi in X:
#            print('New x ------------------')
            p = self.dtree.passon_predict(xi)
            preds.append(p)
        return np.array(preds)

    # def predict_multiprocess(self, X, n_processes = 5):
    #     # TODO: Is this even a good idea, or does the 
    #     # tree get messed up?
    #     with multiprocessing.Pool(n_processes) as pool:
    #         preds = pool.map(self.dtree.passon_predict, X)
    #     return np.array(preds)





class Node:
    ''' Decision tree node, including recursively generated subnodes.
        The tree is lazy, meaning new subnodes are only created when 
        needed.

    # Notes:
    ## left is True
    # right is False
    '''


    def __init__(self, x, y, 
                 criterion,
                 min_leaf_size = 5,
                 jit = True
                 ):
#        print('New node')
        self.x = x 
        self.y = y

        self.criterion = criterion
        self.jit = jit

        self.min_leaf_size = min_leaf_size

        self.set_split_rule()

        self._left = None 
        self._right = None

    def __repr__(self):
        return f"{'Leaf' if self.is_leaf() else 'Decision'}node with size {self.x.shape[0]}, split at {round(self.split_value,3)} in column {self.split_col}."

    def __str__(self):
        if self.is_leaf():
            return f'Leafnode yhat={self.predict()}'
        return f'Decisionnode x[{self.split_col}]<{round(self.split_value,3)}'

    @property
    def left(self):
        ''' Left subnode '''
        if self.is_leaf():
            raise KeyError('Leafs dont have a left subnode')
        if self._left is None:
            self._left = Node(x = self.x[self.leftidx], y = self.y[self.leftidx], criterion = self.criterion)
        return self._left    

    @property 
    def right(self):
        ''' Right subnode '''
        if self.is_leaf():
            raise KeyError('Leafs dont have a right subnode')
        if self._right is None:
            self._right = Node(x = self.x[self.rightidx], y = self.y[self.rightidx], criterion = self.criterion)
        return self._right

    def search_row(self, idx, row):
        ''' Search a row for the best split.
        '''
        if self.jit:
            return search_row_jitted(self.criterion, self.y, idx, row)
        return self._search_row(idx, row)

    def _search_row(self, idx, row):
        score = float('inf')

        for obs in row:
            new_rule = row < obs
            y_left = self.y[new_rule]
            y_right = self.y[~new_rule]

            if len(y_left) > 0:
                score_left = self.criterion(y_left, np.mean(y_left))
            else: 
                score_left = 0
            if len(y_right) > 0:
                score_right = self.criterion(y_right, np.mean(y_right))
            else: 
                score_right = 0
            new_score = -(score_left + score_right)

            if new_score < score:
                score = new_score
                rule = new_rule
                split_col = idx
                split_value = obs

        return score, rule, split_col, split_value
            

    def set_split_rule(self):
        ''' Minimize total MSE in the
            two subleafs.
        '''
        # This needs to go to a faster language
        if self.x.ndim == 1:
            score, rule, split_col, split_value = self.search_row(-1, self.x)
        else:
            score = float('inf')
            for idx, row in enumerate(self.x.T):
                # If there are only one value in
                # the row, we cant use this for 
                # splitting. TODO: could this be
                # a potential stopping point?
                if len(np.unique(row)) <= 1:
                    continue

                new_score, new_rule, new_split_col, new_split_value = self.search_row(idx, row)

                if new_score < score:
                    score = new_score
                    rule = new_rule
                    split_col = new_split_col
                    split_value = new_split_value

        self.rule = rule
        self.leftidx = rule
        self.rightidx = ~rule

        self.split_col = split_col
        self.split_value = split_value


    def is_leaf(self):
        ''' Is this node a leaf? '''
        return self.x.shape[0] < self.min_leaf_size


    def passon_predict(self, xi):
        ''' Predict if the node is a leaf,
            otherwise pass the observaation on
            to a lower node and repeat.
        '''
        if self.is_leaf():
            return self.predict()

        if self.passon_left(xi):
            return self.left.passon_predict(xi)
        return self.right.passon_predict(xi)


    def passon_left(self, x):
        ''' Should x be passed on to the left subnode? '''
        if x.ndim == 1:
            return x[self.split_col] < self.split_value
        return x[:,self.split_col] < self.split_value


    def predict(self):
        ''' Mean response within node '''
        return np.mean(self.y)
