from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

class TOTree:
    
    def __init__(self, 
                 base_estimator = DecisionTreeRegressor(),
                 **grid_params
                 ):
        self.model = None 
        self.base = base_estimator
        self.grid_params = grid_params
        
    def fit(self, X, y, t):
        p = t.mean()
        y = t * (y / p) - (1 - t) * (y / (1 - p))
                        
        grid = GridSearchCV(self.base, self.grid_params, cv = 10)
        grid.fit(X,y)
        
        self.model = grid.best_estimator_
        return self
    
    def predict(self, X):
        return self.model.predict(X)



