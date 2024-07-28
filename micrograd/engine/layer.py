from .neuron import Neuron

class Layer:
    def __init__(self, input_features, output_features, activation=None):
        self.neurons = [Neuron(input_features) for _ in range(output_features)]
        self.activation = activation if activation is not None else self.linear_activation
        
    def __call__(self, x):
        out = [self.activation(n(x)) for n in self.neurons]
        return out[0] if len(out) == 1 else out


    def linear_activation(self, x):
        return x