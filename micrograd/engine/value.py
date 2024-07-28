import math

class Value:

    def __init__(self, value, parents=()):
        self.value = value
        self.parents = parents
        self.grad = 0.0

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        child = Value(self.value + other.value, parents=(self, other, '+'))
        
        def _backward():
            self.grad += 1.0 * child.grad
            other.grad += 1.0 * child.grad

            self._backward()
            other._backward()
        
        child._backward = _backward

        return child

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        child = Value(self.value * other.value, parents=(self, other, '*'))
        
        def _backward():
            self.grad += other.value * child.grad
            other.grad += self.value * child.grad

            self._backward()
            other._backward()

        child._backward = _backward
        
        return child

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __pow__(self, other):
        child = Value(self.value ** other, parents=(self, Value(other), '**'))

        def _backward():
            self.grad += other * (self.value ** (other - 1) * child.grad)
            self._backward()
        
        child._backward = _backward

        return child

    def log(self):
        if self.value <= 0:
            self.value = 1e-17 #Not great but temporary solution to having values of 0.0
        child = Value(math.log(self.value), parents=(self,))

        def _backward():
            self.grad += (1.0 / self.value) * child.grad
            self._backward()

        child._backward = _backward

        return child

    def exp(self):
        child = Value(math.exp(self.value), parents=(self,))

        def _backward():
            self.grad += math.exp(self.value) * child.grad
            self._backward()

        child._backward = _backward

        return child


    def _backward(self):
        pass

    def backward(self):
        self._backward()

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
            if not self.parents:
                parent_info = "None"
            elif len(self.parents) == 2:
                parent_info = f"({self.parents[0].value} {self.parents[1]})"
            elif len(self.parents) == 3:
                parent_info = f"({self.parents[0].value} {self.parents[2]} {self.parents[1].value})"
            else:
                parent_info = str(self.parents)
            return f"Val(value={self.value:.4f}, grad={self.grad:.4f}, parents={parent_info})"