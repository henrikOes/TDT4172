import numpy as np

class LogisticalRegression():
    
    def __init__(self, learning_rate = 0.01, epochs = 1000): 
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []
    
    def sigmoid_function(self, X):
        return 1/(1+np.exp(-X))
        
    def _compute_loss(self, y, y_pred):
        eps = 1e-9
        return  -np.mean(y*np.log(y_pred+eps)+(1-y)*np.log(1-y_pred+eps)) #log-likelihood
        
    def compute_gradients(self, X, y, y_pred):
        grad_w = (1/X.shape[0]) * np.dot(X.T, (y_pred-y)) 
        grad_b = (1/X.shape[0]) * np.sum(y_pred-y)
        return grad_w, grad_b
    
    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b
    
    def accuracy(self, true_values, predictions):
        return np.mean(true_values == predictions)
        
    
    def fit(self, X, y):
        print(">>> fit() called, starting training")
        X = np.array(X)
        features = X.shape[1]
        self.weights = np.zeros(features)
        self.bias = 0
        
        for _ in range(self.epochs):
            lin_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid_function(lin_model)
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)
            
            loss = self._compute_loss(y, y_pred)
            pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred]
            self.train_accuracies.append(self.accuracy(y, pred_to_class))
            
    def predict(self, X):
        lin_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid_function(lin_model)
        return [1 if _y > 0.5 else 0 for _y in y_pred]





