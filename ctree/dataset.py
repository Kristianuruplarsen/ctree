
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, 
                 homogeneous = True,
                 expected_effect = 3,
                 p_treated = 0.5,
                 n = 1000, 
                 m = 1,
                 seed = 1,
                 ):
        np.random.seed(seed)
        self.expected_effect = expected_effect
        self.p_treated = p_treated
        
        self.X = np.random.normal(size = (n, m))
        self.b = np.random.normal(size = m)

        self.t = (np.random.uniform(size = n) < p_treated).astype(int)
        self.eps = np.random.normal(size = n)

        if homogeneous:
            self.effect = expected_effect
        else:
            self.effect = self.tau(self.X[:,0])
        # elif random:
        #     self.effect = np.random.normal(loc = expected_effect, size = n)

        self.y0 = self.X.dot(self.b) + self.eps
        self.y1 = self.y0 + self.effect

        self.y = self.t * self.y1 +(1 - self.t) * self.y0

    @property 
    def df(self):
        data = {
            'y': self.y,
            'y0': self.y0,
            'y1': self.y1, 
            't': self.t,
            'effect': self.effect,
            'epsilon': self.eps,
            **{
                f'X{i}': self.X[:,i] for i in range(self.X.shape[1])
            }
        }
        return pd.DataFrame(data)


    def tau(self, x):
        result = 2 * self.expected_effect / (1 + np.exp(-x))
        return result


def RCT_train_test_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)

    if 'effect' in X.columns:
        true_train = X_train['effect']
        true_test = X_test['effect']    
    else:
        true_train, true_test = None, None

    T_train = X_train['t']
    T_test = X_test['t']

    X_train = X_train[[i for i in X_train.columns if i[0] == 'X']]
    X_test = X_test[[i for i in X_test.columns if i[0] == 'X']]    
        
    return (X_train, X_test), (y_train, y_test), (T_train, T_test), (true_train, true_test)
