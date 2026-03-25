import math


class Weight:

    def __init__(self, data, label = 'ukn', _children = [], _op = None) -> None:
        self.data = data
        self.label = label
        self.grad = 0.0
        self._children = _children
        self._op = _op
        self._backward = lambda : None

    def __repr__(self) -> str:
        return self.label + "|" + str(self.data) + "|" + str(self.grad)
    
    def __add__(self, other):
        other = other if isinstance(other, Weight) else Weight(other)
        out = Weight(self.data + other.data, _children=[self, other], _op="+")

        def backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Weight) else Weight(other)
        out = Weight(self.data * other.data, _children=[self, other], _op="*")

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward

        return out
    
    def __neg__(self): # -self
        return self * -1
    
    def __sub__(self, other):
        return self + other * (-1)
    
    def __pow__(self, other):
        other = other if isinstance(other, Weight) else Weight(other)
        out = Weight(self.data ** other.data, _children=[self, other], _op="pow")

        def backward():
            self.grad += other.data * self.data**(other.data - 1) * out.grad

        out._backward = backward

        return out
    
    def __truediv__(self, other):
        return self * other**(-1)
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __rsub__(self, other): # other - self
        return other + (-self)
    
    def __rmul__(self, other): # other * self
        return self * other
    
    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def ReLU(self):
        out = Weight(max(0, self.data), _children=[self], _op="ReLU")

        def backward():
            grad = 1 if self.data > 0 else 0  # if x = 0 choose subgradient 0
            self.grad += grad * out.grad

        out._backward = backward

        return out

    def sigmoid(self):
        e = math.exp(self.data)
        out = Weight(e / (1 + e), _children=[self], _op="Sigmoid")

        def backward():
            grad = (e / (1 + e)) * (1 - e / (1 + e))
            self.grad += grad * out.grad

        out._backward = backward

        return out

    def log(self):
        out = Weight(math.log(self.data), _children=[self], _op="ln")

        def backward():
            self.grad += 1 / self.data * out.grad

        out._backward = backward

        return out
    
    def children(self):
        return self._children
    
    def backward(self):
        """Backprop uses reverse topological order"""
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children():
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
