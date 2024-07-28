from micrograd.engine.value import Value

class MSE:

    def __call__(self, y_pred, y):
        loss = Value(0.0)
        
        n = len(y_pred)

        for i in range(n):
            loss += (y_pred[i] - y[i]) ** 2

        loss = loss * (Value(0.5) * Value(1/n))
        loss.grad = 1.0
        
        return loss