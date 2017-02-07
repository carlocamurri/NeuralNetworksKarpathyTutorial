# Simplified version of multiplication gate to return the output or the gradient of the input variables
def multiply_gate(a, b, dx=1, backwards_pass=False):
    if not backwards_pass:
        return a * b
    else:
        da = b * dx
        db = a * dx
        return da, db


def add_gate(a, b, dx=1, backwards_pass=False):
    if not backwards_pass:
        return a + b
    else:
        da = 1 * dx
        db = 1 * dx
        return da, db


# We can combine different gates together
# For example, a + b + c
def add_gate_combined(a, b, c, dx=1, backwards_pass=False):
    if not backwards_pass:
        return a + b + c
    else:
        da = 1 * dx
        db = 1 * dx
        dc = 1 * dx
        return da, db, dc


# Another example: combining addition and multiplication (a * b + c)
def add_multiplication_combined(a, b, c, dx=1, backwards_pass=False):
    if not backwards_pass:
        return a * b + c
    else:
        da = b * dx
        db = a * dx
        dc = 1 * dx
        return da, db, dc


def sigmoid(x):
    return 1 / (1 + (exp(-x)))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# An even more complex neuron (sigmoid(a*x + b*y + c))
def complex_neuron(a, b, c, x, y, df=1, backwards_pass=False):
    if not backwards_pass:
        q = a*x + b*y + c
        f = sigmoid(q)
        return f
    else:
        dq = sigmoid_derivative(f) * df
        da = x * dq
        dx = a * dq
        dy = b * dq
        db = y * dq
        dc = 1 * dq
        return da, db, dc, dx, dy


# What if both inputs of a multiplication are equal (a * a)?
def square_neuron(a, dx=1, backwards_pass=False):
    if not backwards_pass:
        return a * a
    else:
        # From power rule:
        da = 2 * a * dx
        # Short form for:
        # da = a * dx
        # da += a * dx
        return da


# For a*a + b*b + c*c:
def sum_squares_neuron(a, b, c, dx=1, backwards_pass=False):
    if not backwards_pass:
        return a*a + b*b + c*c
    else:
        da = 2 * a * dx
        db = 2 * b * dx
        dc = 2 * c * dx