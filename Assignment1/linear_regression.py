import numpy as np

class LinearRegression():
    
    def __init__(self): 
        self.weights, self.bias = None, None
        self.epochs = 1000
        self.learning_rate = 0.1
        self.losses, self.train_accuracies = [], []
        pass
    
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        self.weights = np.zeros(x.shape[1])
        self.bias = 0
        
        for _ in self.epochs
        
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        # TODO: Implement
        raise NotImplementedError("The predict method is not implemented yet.")





