import random
from micrograd.engine import Weight


class Neuron:
    """A neuron will have multiple inputs and a single output (activation)"""

    def __init__(self, in_features, activation_fn = 'linear') -> None:
        """activation function expects 'linear', 'relu' or 'sigmoid'"""
        self.in_features = in_features
        self.activation_fn = activation_fn
        self.w = [Weight(random.random(), 'w') for _ in range(self.in_features)]
        self.b = Weight(random.random(), 'b')

    def __call__(self, x):
        assert len(x) == len(self.w), "Dimension mismatch"
        activation = sum([self.w[i] * x[i] for i in range(self.in_features)]) + self.b
        if self.activation_fn == 'linear':
            return activation
        elif self.activation_fn == 'relu':
            return activation.ReLU()
        elif self.activation_fn == 'sigmoid':
            return activation.sigmoid()
    
    def parameters(self):
        return self.w + [self.b]


class Layer:
    """A layer of n Neurons"""

    def __init__(self, in_features, out_features, activation_fn = 'linear') -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.neurons = [Neuron(self.in_features, self.activation_fn) for _ in range(self.out_features)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class NeuralNet:
    """A Linear neural network of n Layers"""

    def __init__(self, in_features, out_features : list, activation_fn : list) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.input_dim = [self.in_features] + self.out_features[:-1]
        self.output_dim = self.out_features
        self.layers = [Layer(self.input_dim[i], self.output_dim[i], self.activation_fn[i]) for i in range(len(self.input_dim))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for n in self.layers for p in n.parameters()]