from micrograd.engine.value import Value
from micrograd.engine.layer import Layer
from micrograd.engine.model import Model

from micrograd.nets.NeuralNet import NeuralNet

from micrograd.activation_functions.activations import sigmoid_activation

class ComplexLogisticRegression(NeuralNet):

    def __init__(self, image_size):
        super().__init__()
        # Set-up a fully-connected neural net for training
        layers = [
            Layer(image_size, 5, activation="relu_activation"),
            Layer(5, 3, activation="relu_activation"),
            Layer(3, 1, activation="sigmoid_activation")
        ]

        self.model = Model(layers)