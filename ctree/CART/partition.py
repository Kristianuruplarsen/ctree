
import numpy as np 
from numba import njit

from .node import Node


def SSE(y, yhat):
    return np.sum((y - yhat)**2)

def SAE(y, yhat):
    return np.sum(np.abs(y - yhat))

def HonestSSE(y, yhat):
    return np.sum((y - yhat)**2 - y**2)


def meanpredictor(y):
    return np.mean(y)

def medianpredictor(y):
    return np.median(y)


class Partition:
    ''' This class wraps the CART Node
        for abstraction.
    '''
    _CRITERIA = {'mse': SSE,
                 'mae': SAE,
                 'honest': HonestSSE}

    _PREDICTORS = {'mean': meanpredictor,
                   'median': medianpredictor}

    def __init__(self,
                 criterion = 'mse', 
                 predictor = 'mean',
                 jit = True,
                 min_leaf_size = 5
                ):

        self.criterion_str = criterion
        self.predictor_str = predictor
        self.jit = jit

        self.criterion = self._CRITERIA[criterion]
        self.predictor = self._PREDICTORS[predictor]

        if jit:
            self.criterion = njit(self.criterion)
            self.predictor = njit(self.predictor)

        self.min_leaf_size = min_leaf_size
        self.dtree = None


    def fit(self, X, y):

        self.dtree = Node(X, y,
                          jit = self.jit, 
                          criterion=self.criterion, 
                          predictor = self.predictor,
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


