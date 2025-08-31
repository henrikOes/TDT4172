import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate = 0.001, epochs = 1000): 
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
    
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        X = np.array(X).reshape(-1, 1)  # ensures 2d

        samples, features = X.shape 
        self.weights = np.zeros(features)
        
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            
            d_w = (1/samples) * np.dot(X.T, (y_pred-y)) 
            d_b = (1/samples) * np.sum(y_pred-y)
            
            self.weights -= self.learning_rate * d_w
            self.bias -= self.learning_rate * d_b
        
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
        X = np.array(X).reshape(-1, 1) 
        return np.dot(X, self.weights) + self.bias





