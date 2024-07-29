from pydantic import BaseModel, Field
from typing import List

# Weights
class NeuronSchema(BaseModel):
    values: List[float] = Field(description="Weights for each neuron in the layer, including bias")

class LayerSchema(BaseModel):
    input_features: int = Field(description="Number of input features for this layer")
    output_features: int = Field(description="Number of output features for this layer")
    activation: str = Field(default="linear_activation", description="Activation function used in this layer")
    neurons: List[NeuronSchema] = Field(description="Weights for each neuron in the layer, including bias")

class ModelSchema(BaseModel):
    layers: List[LayerSchema] = Field(description="Weights for each layer in the model")