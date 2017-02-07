from __future__ import division
from __future__ import print_function

# Initialize some data
X_test = [[1.2, 0.7], [-0.3, 0.5], [3, 2.5]]
y_test = [1, -1, 1]  # Classes
w_test = [0.1, 0.2, 0.3]  # Example, random numbers
# Setting regularization parameter
alpha = 0.1


def cost(X, y, w):
    total_cost = 0
    m = len(X)
    n = len(X[0])
    for i in range(0, m):
        score = 0
        for j in range(0, n):
            score += X[i][j] * w[j]
        score += w[n]
        cost_i = max(0, -(y[i]) * score + 1)
        print("Example:", i, "\tScore computed:", score, "\tCost computed:", cost_i)
        total_cost += cost_i
    # Regularization cost:
    reg_cost = 0
    for k in range(0, n+1):
        reg_cost += w[k]**2
    reg_cost *= alpha
    print("Regularization cost for current model:", reg_cost)
    total_cost += reg_cost
    print("Total cost is", total_cost)
    return total_cost

cost(X_test, y_test, w_test)