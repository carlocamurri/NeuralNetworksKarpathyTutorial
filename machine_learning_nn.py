from __future__ import print_function
from __future__ import division
import random
import math


class Unit:
    def __init__(self, value, grad):
        self.value = value
        self.grad = grad


class MultiplyGate:
    def __init__(self):
        self.u0 = None
        self.u1 = None
        self.output = None

    def forward(self, u0, u1):
        self.u0 = u0
        self.u1 = u1
        self.output = Unit(u0.value * u1.value, 0)
        return self.output

    def backward(self):
        self.u0.grad += self.u1.value * self.output.grad
        self.u1.grad += self.u0.value * self.output.grad


class AddGate:
    def __init__(self):
        self.u0 = None
        self.u1 = None
        self.output = None

    def forward(self, u0, u1):
        self.u0 = u0
        self.u1 = u1
        self.output = Unit(u0.value + u1.value, 0)
        return self.output

    def backward(self):
        self.u0.grad += 1 * self.output.grad
        self.u1.grad += 1 * self.output.grad


class Circuit:
    def __init__(self):
        self.m0 = MultiplyGate()
        self.m1 = MultiplyGate()
        self.a0 = AddGate()
        self.a1 = AddGate()
        self.ax = None
        self.by = None
        self.ax_by = None
        self.ax_by_c = None

    def forward(self, x, y, a, b, c):
        self.ax = self.m0.forward(a, x)
        self.by = self.m1.forward(b, y)
        self.ax_by = self.a0.forward(self.ax, self.by)
        self.ax_by_c = self.a1.forward(self.ax_by, c)
        return self.ax_by_c

    def backward(self, gradient_top):
        self.ax_by_c.grad = gradient_top
        self.a1.backward()
        self.a0.backward()
        self.m1.backward()
        self.m0.backward()


# Support Vector Machine class
class SVM:
    def __init__(self):
        # Initializing with random values (make sure gradient remains 0)
        self.a = Unit(1, 0)
        self.b = Unit(-2, 0)
        self.c = Unit(-1, 0)
        self.circuit = Circuit()
        self.unit_out = None

    def forward(self, x, y):
        self.unit_out = self.circuit.forward(x, y, self.a, self.b, self.c)
        return self.unit_out

    def backward(self, label):
        # Reset gradients on a, b, and c
        self.a.grad = 0
        self.b.grad = 0
        self.c.grad = 0
        pull = label - self.unit_out.value
        # Compute pull based on circuit output
        # pull = 0
        # if label == 1 and self.unit_out.value < 1:
        #    pull = 1  # Score was too low, pull up
        # elif label == -1 and self.unit_out.value > -1:
        #    pull = -1  # Score was too high, pull down
        self.circuit.backward(pull)  # Writes gradients into a, b, c, x, y
        # Add regularization proportional to value of a and b (not on constant term)
        self.a.grad -= self.a.value
        self.b.grad -= self.b.value

    def learn_from(self, x, y, label):
        self.forward(x, y)  # Forward pass (set values in all units)
        self.backward(label)  # Backward pass (set gradient in all units)
        self.parameter_update()  # Have parameters respond according to values calculated

    def parameter_update(self):
        step_size = 0.01
        self.a.value += step_size * self.a.grad
        self.b.value += step_size * self.b.grad
        self.c.value += step_size * self.c.grad


# Stochastic Gradient Descent
training_set = [[1.2, 0.7], [-0.3, -0.5], [3.0, 0.1], [-0.1, -1.0], [-1.0, 1.1], [2.1, -3]]
labels = [1, -1, 1, -1, -1, 1]
svm = SVM()


def eval_training_accuracy():
    num_correct = 0
    for i in range(0, len(training_set)):
        x = Unit(training_set[i][0], 0)
        y = Unit(training_set[i][1], 1)
        current_label = labels[i]
        # Check if prediction matches current label
        prediction = svm.forward(x, y).value
        if prediction > 0:
            prediction = 1
        else:
            prediction = -1
        if prediction == current_label:
            num_correct += 1
    avg = num_correct / len(training_set)
    return avg


# Learning loop
for iteration in range(0, 400):
    i = int(math.floor(random.uniform(0, 1) * len(training_set)))
    x1 = Unit(training_set[i][0], 0)
    y1 = Unit(training_set[i][1], 0)
    label1 = labels[i]
    svm.learn_from(x1, y1, label1)

    if iteration % 10 == 0:
        print("Training set accuracy at iteration", iteration, ":", eval_training_accuracy())


# Simplified SVM
# Initial parameters (random)
a = 1
b = -2
c = -1
for iteration in range(0, 400):
    # Pick a random data point
    i = int(math.floor(random.uniform(0, 1) * len(training_set)))
    x1 = training_set[i][0]
    y1 = training_set[i][1]
    label1 = labels[i]
    # Compute pull
    score = a * x1 + b * y1 + c
    pull = label1 - score
    # Compute gradient and update parameters
    step_size = 0.01
    a += step_size * (x1 * pull - a)  # The -a is from regularization
    b += step_size * (y1 * pull - b)  # The -b is from regularization
    c += step_size * (1 * pull)
