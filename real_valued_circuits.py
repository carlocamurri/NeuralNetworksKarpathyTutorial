from __future__ import division

import random

x_in = -2.0
y_in = 3.0


def forward_multiply_gate(x, y):
    return x * y


print forward_multiply_gate(-2, 3)


# How to "tweak" the input slightly to increase the output?

# Strategy #1: Random Local Search

def random_local_search(x, y):
    tweak_amount = 0.01
    best_out = -9999999
    best_x = x
    best_y = y
    for i in range(100):
        x_try = x + tweak_amount * (random.uniform(0, 1) * 2 - 1)
        y_try = y + tweak_amount * (random.uniform(0, 1) * 2 - 1)
        out = forward_multiply_gate(x_try, y_try)
        if out > best_out:
            best_out = out
            best_x = x_try
            best_y = y_try
    return best_out, best_x, best_y


print random_local_search(-2, 3)


# Strategy #2: Numerical Gradient
def x_derivative(x, y, h):
    xph = x + h
    out = forward_multiply_gate(x, y)
    out2 = forward_multiply_gate(xph, y)
    return (out2 - out) / h


def y_derivative(x, y, h):
    yph = y + h
    out = forward_multiply_gate(x, y)
    out2 = forward_multiply_gate(x, yph)
    return (out2 - out) / h


x_deriv = x_derivative(x_in, y_in, 0.0001)
y_deriv = y_derivative(x_in, y_in, 0.0001)
print x_deriv, y_deriv


def increment_multiply_gate(x, y, step_size):
    out = forward_multiply_gate(x, y)
    x = x + step_size * x_derivative(x, y, 0.0001)
    y = y + step_size * y_derivative(x, y, 0.0001)
    return forward_multiply_gate(x, y)


print increment_multiply_gate(x_in, y_in, 0.01)
print increment_multiply_gate(x_in, y_in, 1)
print increment_multiply_gate(x_in, y_in, 10)


# Strategy #3: Analytic gradient
# The gradient of f(x, y) with respect to x (in this case) is y (using calculus)
# The gradient of f(x, y) with respect to y (in this case) is x (using calculus)
# This returns an exact gradient
def multiply_analytical_gradient(x, y):
    x_deriv = y
    y_deriv = x
    return x_deriv, y_deriv

