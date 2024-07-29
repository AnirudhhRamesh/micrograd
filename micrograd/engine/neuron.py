import random
from .value import Value
from micrograd.schemas.schemas import NeuronSchema

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

    def export(self):
        return NeuronSchema(values=([w.value for w in self.w] + [self.b.value]))

    @classmethod
    def from_json(cls, neuron: NeuronSchema):
        w = [Value(v) for v in neuron.values[:-1]]
        b = Value(neuron.values[-1])
        
        #TODO Not great, but allows to keep simple Neuron constructor
        neuron = cls(len(neuron.values)-1)
        neuron.w = w
        neuron.b = b

        return neuron