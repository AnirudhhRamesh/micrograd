import random

class SimpleOptimizer:
    def __init__(self, parameters, learn_rate, mode='default') -> None:
        self.parameters = parameters
        self.learn_rate = learn_rate
        self.set_mode(mode)

    def set_mode(self, mode):
        modes={
            "default":self.simple_step,
            "SGD":self.SGD_step
        }

        self.step_mode = modes.get(mode)

    def step(self):
        self.step_mode()

    def simple_step(self):
        for p in self.parameters:
            p.value = p.value - self.learn_rate * p.grad

    def SGD_step(self):
        # Only update a small set of parameters
        sample_size = max(1, len(self.parameters) // 50)
        params = random.sample(self.parameters, sample_size)
        
        for p in params:
            p.value = p.value - self.learn_rate * p.grad

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0

    def set_learn_rate(self, lr):
        self.learn_rate = lr

    def get_learn_rate(self):
        return self.learn_rate