from micrograd.engine.layer import Layer
from micrograd.engine.model import Model

class NeuralNet:
    def __init__(self):
        # Default
        self.layers = [Layer(input_features=1, output_features=1)]
        self.model = Model(layers=self.layers)

    def __call__(self, x):
        return self.model(x)

    def parameters(self):
        return self.model.parameters()