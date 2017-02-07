from math import exp


# Let us consider the function: f(x) = sigmoid(a * x + b * y + c)

def sigmoid(x):
    return 1 / (1 + (exp(-x)))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Every 'wire' has 2 numbers associated with it:
#   1. The value it carries during the forward pass
#   2. The gradient that flows back through it in the backward pass
class Unit:
    def __init__(self, value, grad):
        self.value = value
        self.grad = grad


class MultiplyGate:
    def __init__(self):
        self.u0 = None
        self.u1 = None
        self.utop = None

    def forward(self, u0, u1):
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit(u0.value * u1.value, 0.0)
        return self.utop

    # Chain output gradient to local gradients (chain rule from before)
    def backward(self):
        self.u0.grad += self.u1.value * self.utop.grad
        self.u1.grad += self.u0.value * self.utop.grad


class AddGate:
    def __init__(self):
        self.u0 = None
        self.u1 = None
        self.utop = None

    def forward(self, u0, u1):
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit(u0.value + u1.value, 0.0)
        return self.utop

    # Chain output gradient to local gradients (chain rule from before)
    def backward(self):
        self.u0.grad += 1 * self.utop.grad
        self.u1.grad += 1 * self.utop.grad


class SigmoidGate:
    def __init__(self):
        self.u0 = None
        self.utop = None

    def forward(self, u0):
        self.u0 = u0
        self.utop = Unit(sigmoid(u0.value), 0.0)
        return self.utop

    # Chain output gradient to local gradients (chain rule from before)
    def backward(self):
        self.u0.grad += sigmoid_derivative(self.u0.value) * self.utop.grad


# Example input units
a = Unit(1, 0)
b = Unit(2, 0)
c = Unit(-3, 0)
x = Unit(-1, 0)
y = Unit(3, 0)

# Store units in array for future reference
input_units = [a, b, c, x, y]

# Creating the gates
mult_g0 = MultiplyGate()
mult_g1 = MultiplyGate()
add_g0 = AddGate()
add_g1 = AddGate()
sigmoid_g0 = SigmoidGate()

gates = [sigmoid_g0, add_g1, add_g0, mult_g1, mult_g0]

# Define the forward pass
ax = mult_g0.forward(a, x)
by = mult_g1.forward(b, y)
ax_plus_by = add_g0.forward(ax, by)
ax_plus_by_plus_c = add_g1.forward(ax_plus_by, c)
s = sigmoid_g0.forward(ax_plus_by_plus_c)

# Initialize gradient of final output unit to 1 (default initial value)
s.grad = 1.0


def backward_neuron(gates):
    for gate in gates:
        gate.backward()


backward_neuron(gates)


# Let's make the input respond to the computed gradient to check if the function increased
def tug(step_size, input_list):
    for unit in input_list:
        unit.value += step_size * unit.grad
        print "grad: ", unit.grad
        print "value: ", unit.value


tug(0.01, input_units)


# We forward the neuron again to update the output
ax = mult_g0.forward(a, x)
by = mult_g1.forward(b, y)
ax_plus_by = add_g0.forward(ax, by)
ax_plus_by_plus_c = add_g1.forward(ax_plus_by, c)
s = sigmoid_g0.forward(ax_plus_by_plus_c)

print "Circuit output after one backpropagation: ", s.value
