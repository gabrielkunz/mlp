import numpy as np
from perceptron import Perceptron

class MLP:
    def __init__(self, X, i = 2, j = 2, k = 1, alpha = 0.1):
        # X = input data
        # i = number of neurons for the input layer
        # j = number of neurons for the hidden layer
        # k = number of neurons for the output layer
        self.p = Perceptron(X.shape[1], alpha)

        self.input_layer = []
        for index in np.arange(0, i):
            self.input_layer.append(Perceptron(X.shape[1], alpha))

        self.hidden_layer = []
        for index in np.arange(0,j):
            self.hidden_layer.append(Perceptron(X.shape[1], alpha))

        self.output_layer = []
        for index in np.arange(0,k):
            self.output_layer.append(Perceptron(X.shape[1], alpha))

        print("MLP Created.")
        print("Nodes on input layer: {}".format(len(self.input_layer)))
        print("Nodes on hidden layer: {}".format(len(self.hidden_layer)))
        print("Nodes on output layer: {}".format(len(self.output_layer)))

    def derivate(self, x):
        # Sigmoid derivate
        y = x * (1 - x)
        return y

    def fit_single(self, X, target_prediction, epochs = 100):
        # Fit single perceptron
        self.p.fit(X, target_prediction, epochs)
    
    def fit(self, X, target_prediction, epochs = 100):
        # Fit MLP

        # Add bias column
        X = np.c_[X, np.ones((X.shape[0]))]

        # Forward Propagation
        for epoch in np.arange(0, epochs):
            input_layer_Y = []
            hidden_layer_Y = []
            output_layer_Y = []
            error_output_layer = []
            # Input layer
            for index in np.arange(0, len(self.input_layer)):
                # Loop each data input
                for (x, target) in zip(X, target_prediction):
                    weighted_x = np.dot(x, self.input_layer[index].weight)
                    input_layer_Y.append(self.input_layer[index].activation(weighted_x))
                # Convert the output of the layer to match the initial input data type
                self.input_layer_Y = np.array(input_layer_Y)

            # Hidden layer
            for index in np.arange(0, len(self.hidden_layer)):
                # Loop each data input
                for (x, target) in zip(self.input_layer_Y, target_prediction):
                    weighted_x = np.dot(x, self.hidden_layer[index].weight)
                    hidden_layer_Y.append(self.hidden_layer[index].activation(weighted_x))
                # Convert the output of the layer to match the initial input data type
                self.hidden_layer_Y = np.array(hidden_layer_Y)

            # Output layer
            for index in np.arange(0, len(self.output_layer)):
                # Loop each data input
                for (x, target) in zip(self.hidden_layer_Y, target_prediction):
                    weighted_x = np.dot(x, self.output_layer[index].weight)
                    p = self.output_layer[index].activation(weighted_x)
                    output_layer_Y.append(p)
                    error = p - target
                    error_output_layer.append(error[0])
                # Convert the output of the layer to match the initial input data type
                self.output_layer_Y = np.array(output_layer_Y)
                self.error_output_layer = np.array(error_output_layer)

            # Backpropagation
            #self.update_weights(X, target_prediction)
        
        #print("Output from input layer: {}".format(self.input_layer_Y))
        #print("Output from hidden layer: {}".format(self.hidden_layer_Y))
        #print("Output from output layer: {}".format(self.output_layer_Y))
        #print("Error in the output layer: {}".format(self.error_output_layer))

    def update_weights(self, X, target_prediction):
        # Calculate deltas for error gradient and update weights
        # delta_k = y_k(1 - y_k) * e_k
        delta_output_layer = []
        for k in np.arange(len(self.output_layer)):
            delta_output_layer.append(self.derivate(self.output_layer_Y[k]) * self.error_output_layer[k])
        # w_k = w_k - alpha * y_k * e_k
        self.output_layer[k].weight += -self.output_layer[k].alpha * self.output_layer_Y[k] * self.error_output_layer[k]

        # delta_j = y_j(1 - y_j) * sum (w_jk) * delta_k
        delta_hidden_layer = []
        for j in np.arange(0, len(self.hidden_layer)):
            for k in np.arange(0, len(delta_output_layer)):
                sum = self.hidden_layer[j].weight * delta_output_layer[k]
            delta_hidden_layer.append(self.derivate(self.hidden_layer_Y[j]) * sum)

    def predict_single(self, X):
        # Single perceptron predict
        return self.p.predict(X)

    def predict(self, X, add_bias = True):
        # MLP predict
        X = np.atleast_2d(X)

        if add_bias:
            X = np.c_[X, np.ones((X.shape[0]))]

        # Input layer
        for index in np.arange(0, len(self.input_layer)):
            # Loop each data input
            for (x) in zip(X):
                weighted_x = np.dot(x, self.input_layer[index].weight)
                self.input_layer_Y[index] = self.input_layer[index].activation(weighted_x)

        # Hidden layer
        for index in np.arange(0, len(self.hidden_layer)):
            # Loop each data input
            for (x) in zip(X):
                weighted_x = np.dot(x, self.hidden_layer[index].weight)
                self.hidden_layer_Y = self.hidden_layer[index].activation(weighted_x)

        # Output layer
        for index in np.arange(0, len(self.output_layer)):
            # Loop each data input
            for (x) in zip(X):
                weighted_x = np.dot(x, self.output_layer[index].weight)
                prediction = self.output_layer[index].activation(weighted_x)

        return prediction