from __future__ import print_function
from __future__ import division
import random
import math

# Same data as before
training_set = [[1.2, 0.7], [-0.3, -0.5], [3.0, 0.1], [-0.1, -1.0], [-1.0, 1.1], [2.1, -3]]
labels = [1, -1, 1, -1, -1, 1]

# Initialize random parameters
a1 = random.uniform(0, 0.5)
b1 = random.uniform(0, 0.5)
c1 = random.uniform(0, 0.5)
a2 = random.uniform(0, 0.5)
b2 = random.uniform(0, 0.5)
c2 = random.uniform(0, 0.5)
a3 = random.uniform(0, 0.5)
b3 = random.uniform(0, 0.5)
c3 = random.uniform(0, 0.5)
a4 = random.uniform(0, 0.5)
b4 = random.uniform(0, 0.5)
c4 = random.uniform(0, 0.5)
d4 = random.uniform(0, 0.5)

for iteration in range(0, 20000):
    # Pick a random data point
    i = int(math.floor(random.uniform(0, 1) * len(training_set)))
    x = training_set[i][0]
    y = training_set[i][1]
    label = labels[i]
    # Compute forward pass
    # Note: these neurons are also called rectified linear units (ReLU), which use the
    # activation function f(x) = max(0, x) which is also analogous to half-wave rectification in electrical engineering
    n1 = max(0, a1 * x + b1 * y + c1)  # Activation of first hidden neuron
    n2 = max(0, a2 * x + b2 * y + c2)  # Activation of second hidden neuron
    n3 = max(0, a3 * x + b3 * y + c3)  # Activation of third hidden neuron
    output = a4 * n1 + b4 * n2 + c4 * n3 + d4
    # Compute the pull
    pull = label - output
    # Compute backwards pass to all parameters of the model
    # Backpropagation through last "score" neuron
    da4 = n1 * pull
    dn1 = a4 * pull
    db4 = n2 * pull
    dn2 = b4 * pull
    dc4 = n3 * pull
    dn3 = c4 * pull
    dd4 = 1 * pull
    # Backpropagation on the ReLU non-linearities in place
    # i.e. if the neurons did not fire (returned 0) set the gradients to 0
    if n3 == 0:
        dn3 = 0
    if n2 == 0:
        dn2 = 0
    if n1 == 0:
        dn1 = 0
    # Backpropagate to parameters of neuron 1
    da1 = x * dn1
    db1 = y * dn1
    dc1 = 1 * dn1
    # Backpropagate to parameters of neuron 2
    da2 = x * dn2
    db2 = y * dn2
    dc2 = 1 * dn2
    # Backpropagate to parameters of neuron 3
    da3 = x * dn3
    db3 = y * dn3
    dc3 = 1 * dn3
    # No need to backpropagate into x and y as we do not need those gradients
    # Add pulls from regularization (not on the biases) proportional to their value
    da1 -= a1
    db1 -= b1
    da2 -= a2
    db2 -= b2
    da3 -= a3
    db3 -= b3
    da4 -= a4
    db4 -= b4
    dc4 -= c4
    # Finally do the parameter update
    step_size = 0.01
    a1 += step_size * da1
    b1 += step_size * db1
    c1 += step_size * dc1
    a2 += step_size * da2
    b2 += step_size * db2
    c2 += step_size * dc2
    a3 += step_size * da3
    b3 += step_size * db3
    c3 += step_size * dc3
    a4 += step_size * da4
    b4 += step_size * db4
    c4 += step_size * dc4
    d4 += step_size * dd4
    print(d4)

print(d4)
