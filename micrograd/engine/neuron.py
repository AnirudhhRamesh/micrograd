from .value import Value

import random

class Neuron:

    def __init__(self, input_features):
        """
        Initializes a neuron with a w for each feature of a data sample x
        """
        self.w = [Value(random.random()) for _ in range(input_features)]
        self.b = Value(0.0)

    def __call__(self, x):
        """
        Computes the forward pass (i.e. the prediction) given a dataset (matching the initialized model dimensions)
        """
        result = Value(0.0)
        for wi, xi in zip(self.w, x):
            result += wi * xi 

        result += self.b
        
        return result

    def parameters(self):
        return self.w + [self.b]