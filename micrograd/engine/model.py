import json
from micrograd.schemas.schemas import ModelSchema
from typing import List

from .layer import Layer

class Model:
    def __init__(self, layers:List[Layer]) -> None:
        self.layers = layers

    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        return out

    def parameters(self):
        params = []

        for layer in self.layers:
            for neuron in layer.neurons:
                params += neuron.parameters()

        return params
    
    def export(self):
        layers: ModelSchema = []
        for layer in self.layers:
            layers.append(layer.export())
        return ModelSchema(layers=layers)

    @classmethod
    def from_json(cls, model: ModelSchema):
        layers = [
            Layer.from_json(layer) for layer in model.layers
        ]
        return cls(layers)