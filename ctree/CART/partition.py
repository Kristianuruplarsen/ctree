
import numpy as np 
from numba import njit

from .node import Node

def SSE(yhat, y):
    return np.sum((y - yhat)**2)

def SAE(yhat, y):
    return np.sum(np.abs(y - yhat))



class Partition:
    ''' This class wraps the CART Node
        for abstraction.
    '''
    _CRITERIA = {'mse': SSE,
                 'mae': SAE}

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
        if X.ndim == 1:
            p = self.dtree.passon_predict(X)
            return np.array(p)
        for xi in X:
            p = self.dtree.passon_predict(xi)
            preds.append(p)
        return np.array(preds)


