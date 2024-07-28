from micrograd.engine.value import Value
from micrograd.engine.layer import Layer
from micrograd.engine.model import Model

from micrograd.nets.NeuralNet import NeuralNet

class ComplexLogisticRegression(NeuralNet):

    def __init__(self, image_size):
        super().__init__()

        # Set-up a fully-connected neural net for training
        self.layers = [
            Layer(image_size, 3),
            Layer(3, 1, activation=self.logistic_activation)
        ]

        self.model = Model(self.layers)

    # Define the activation function as the logistic activation (1 / (1 - e^-f(x)))
    def logistic_activation(self, x):
        """
        Note: For classification we need the logistic activation to ensure our output is mapped between 0 and 1. 
        This allows us to apply log function without domain errors (required for our Binary Cross entropy loss function).
        """
        # print(f"Test: {x}")
        return Value(1.0) / (Value(1.0) + (-x).exp())