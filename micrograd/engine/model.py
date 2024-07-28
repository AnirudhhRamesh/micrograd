class Model:
    def __init__(self, layers) -> None:
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