class SimpleOptimizer:
    def __init__(self, parameters, learn_rate) -> None:
        self.parameters = parameters
        self.learn_rate = learn_rate

    def step(self):
        for p in self.parameters:
            p.value = p.value - self.learn_rate * p.grad

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0

    def set_learn_rate(self, lr):
        self.learn_rate = lr

    def get_learn_rate(self):
        return self.learn_rate