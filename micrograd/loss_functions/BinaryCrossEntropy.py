from micrograd.engine.value import Value

class BinaryCrossEntropy:

    def __call__(self, y_pred, y):
        loss = Value(0.0)
        n = len(y_pred)

        for i in range(n):
            loss += (-(y[i] * (y_pred[i]).log()) - ((Value(1.0)-y[i])*(Value(1.0)-y_pred[i]).log()))
        
        loss = loss * Value(1/n)
        loss.grad = 1.0
        
        return loss