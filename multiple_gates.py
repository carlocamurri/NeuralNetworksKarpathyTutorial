# Defining the complete gate
def addition_gate(x, y):
    return x + y


def multiplication_gate(q, z):
    return q * z


def forward_circuit(a, b, c):
    return multiplication_gate(addition_gate(a,b), c)


# Initial conditions
x = -2
y = 5
z = -4

q = addition_gate(x, y)
f = multiplication_gate(q, z)

# Gradient of the multiply gate with respect to its inputs
derivative_f_wrt_z = q
derivative_f_wrt_q = z

# Derivatives of the add gate with respect to its inputs
derivative_q_wrt_x = 1.0
derivative_q_wrt_y = 1.0

# By the chain rule:
derivative_f_wrt_x = derivative_q_wrt_x * derivative_f_wrt_q
derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q

print derivative_f_wrt_x
print derivative_f_wrt_y

# Final gradient (vector) from above calculations
gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]

print gradient_f_wrt_xyz


# Manipulating inputs by a step size, as we did before
step_size = 0.01
x_new = x + step_size * gradient_f_wrt_xyz[0]
y_new = y + step_size * gradient_f_wrt_xyz[1]
z_new = z + step_size * gradient_f_wrt_xyz[2]

# Update gate values
q_new = addition_gate(x_new, y_new)
f_new = multiplication_gate(q_new, z_new)

print x_new, y_new, z_new
print f_new


# Using numerical gradient check to make sure the analytical gradient is correct
h = 0.0001
x_numerical_derivative = (forward_circuit(x + h, y, z) - forward_circuit(x, y, z)) / h
y_numerical_derivative = (forward_circuit(x, y + h, z) - forward_circuit(x, y, z)) / h
z_numerical_derivative = (forward_circuit(x, y, z + h) - forward_circuit(x, y, z)) / h

print x_numerical_derivative, y_numerical_derivative, z_numerical_derivative