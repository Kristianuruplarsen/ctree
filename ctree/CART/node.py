
import numpy as np

from numba import njit
from .jitfuncs import search_row_jit, search_row_nojit


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
                 predictor,
                 min_leaf_size = 5,
                 jit = True
                 ):
        '''
        '''
        self.x = x 
        self.y = y
        
        self.criterion = criterion
        self.predictor = predictor
        self.jit = jit

        self.min_leaf_size = min_leaf_size

        self._left = None 
        self._right = None
        self.split_col = None 
        self.split_value = None
        self._is_leaf = False 

        self.set_split_rule()

    def __repr__(self):
        s = "{}node with size {}, {}".format(
            'Leaf' if self.is_leaf() else 'Decision',
            self.x.shape[0],
            f'Predicting {self.predict()}' if self.is_leaf() else f'split at {round(self.split_value,3)} in column {self.split_col}.'
        )
        return s 

    def __str__(self):
        if self.is_leaf():
            return f'Leafnode yhat={self.predict()}'
        return f'Decisionnode x[{self.split_col}]<{round(self.split_value,3)}'


    def subnodeopts(self, side = 'left'):
        shared = {'criterion': self.criterion,
                  'predictor': self.predictor,
                  'min_leaf_size': self.min_leaf_size,
                  'jit': self.jit}

        if side == 'left':
            return {'x': self.x[self.leftidx],
                    'y': self.y[self.leftidx],
                    **shared}
        elif side == 'right':
            return {'x': self.x[self.rightidx],
                    'y': self.y[self.rightidx],
                    **shared}

    @property
    def left(self):
        ''' Left subnode '''
        if self.is_leaf():
            raise KeyError('Leafs dont have a left subnode')
        if self._left is None:
            self._left = Node(**self.subnodeopts('left'))
        return self._left    

    @property 
    def right(self):
        ''' Right subnode '''
        if self.is_leaf():
            raise KeyError('Leafs dont have a right subnode')
        if self._right is None:
            self._right = Node(**self.subnodeopts('right'))
        return self._right


    def search_row(self, row):
        ''' Search a row for the best split.
        '''
        if self.jit:
            return search_row_jit(criterion = self.criterion, 
                                  predictor = self.predictor,
                                  y = self.y, 
                                  row = row, 
                                  min_leaf_size = self.min_leaf_size)
                                  
        return search_row_nojit(criterion = self.criterion, 
                                  predictor = self.predictor,
                                  y = self.y, 
                                  row = row, 
                                  min_leaf_size = self.min_leaf_size)


    def set_split_rule(self):
        '''
        '''
        if self.x.ndim == 1:
            # if there is only one column in X we 
            # only need to search this
            score, rule, split_value, leaf = self.search_row(self.x)
            split_col = 0
        else:
            # Otherwise we need to search each
            # column at a time.
            score = np.inf
            leaf = True
            for idx, row in enumerate(self.x.T):
                new_score, new_rule, new_split_value, leaf_candidate = self.search_row(row)
                new_split_col = idx

                if leaf_candidate:
                    continue 
                
                if new_score < score:
                    score = new_score
                    rule = new_rule
                    split_col = new_split_col
                    split_value = new_split_value
                    leaf = False

        if leaf:
            self._is_leaf = True

        else:
            self.rule = rule
            self.leftidx = rule
            self.rightidx = np.logical_not(rule)
            self.split_col = split_col
            self.split_value = split_value


    def is_leaf(self):
        ''' Is this node a leaf? '''
        return self._is_leaf
#        return self.x.shape[0] < self.min_leaf_size


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
        return self.predictor(self.y)
