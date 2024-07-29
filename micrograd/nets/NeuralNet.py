from micrograd.engine.layer import Layer
from micrograd.engine.model import Model
from micrograd.schemas.schemas import ModelSchema

class NeuralNet:
    def __init__(self, model=None):
        self.model = model if model is not None else Model([Layer(1,1)])

    def __call__(self, x):
        return self.model(x)

    def parameters(self):
        return self.model.parameters()

    def export(self):
        print("Exporting model...")
        return self.model.export()

    @classmethod
    def from_json(cls, model:ModelSchema):
        print("Importing model...")
        model = Model.from_json(model)
        print("Imported model!")
        
        return cls(model)