from micrograd.nets.NeuralNet import NeuralNet

from micrograd.engine.layer import Layer
from micrograd.engine.model import Model

class SimpleLinearRegression(NeuralNet):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [
            Layer(input_features=1, output_features=12),
            Layer(12, 1)
        ]
        self.model = Model(layers=self.layers)