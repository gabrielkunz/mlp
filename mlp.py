import numpy as np
from perceptron import Perceptron

class MLP:
    def __init__(self, X, hidden_layers = 1, alpha = 0.1):
        self.p = Perceptron(X.shape[1], alpha)

    def fit(self, X, target_prediction, epochs = 100):
        # Perform the fit outside Perceptron class since the epoch is no applied on
        # the entire net and the error does not come from a single neuron

        # Add bias column
        X = np.c_[X, np.ones((X.shape[0]))]

        # Loop the epochs
        for epoch in np.arange(0, epochs):
            # Loop each data input
            for (x, target) in zip(X, target_prediction):
                # Weighted input
                weighted_x = np.dot(x, self.p.weight)

                # Get the prediction
                p = self.p.activation(weighted_x)

                # Update the weight based on the error, if the error is 0, the weight will remain the same
                if p != target:
                    error = p - target
                    self.p.weight += -self.p.alpha * x * error

    def predict(self, X):
       return self.p.predict(X)
