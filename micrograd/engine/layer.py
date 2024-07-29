from micrograd.activation_functions.activations import linear_activation, relu_activation, sigmoid_activation
from .neuron import Neuron
from micrograd.schemas.schemas import LayerSchema
from typing import List

class Layer:
    def __init__(self, input_features, output_features, neurons: List[Neuron] = None, activation="linear_activation"):
        self.input_features = input_features
        self.output_features = output_features
        self.neurons = neurons if neurons is not None else [Neuron(input_features) for _ in range(output_features)]
        self._set_activation(activation)
        
    def __call__(self, x):
        out = [self.activation(n(x)) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def _set_activation(self, activation:str):
        activations={
            "linear_activation": linear_activation,
            "relu_activation": relu_activation,
            "sigmoid_activation": sigmoid_activation
        }
        self.activation = activations[activation]

    def export(self):
        neurons = []
        for neuron in self.neurons:
            neurons.append(neuron.export())
        
        return LayerSchema(
            input_features=self.input_features,
            output_features=self.output_features,
            activation=self.activation.__name__,
            neurons=neurons
        )

    @classmethod
    def from_json(cls, layer: LayerSchema):
        neurons = [Neuron.from_json(neuron) for neuron in layer.neurons]
        return cls(layer.input_features, layer.output_features, neurons, layer.activation)