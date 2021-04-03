import numpy as np
import matplotlib.pyplot as plt


def cost(X, y, theta):
    return sum((X @ theta - y) ** 2) / (2 * X.shape[0])


def gradientDescent(X, y, theta, alpha, num_iters):
    m = X.shape[0]
    j_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        theta = theta - (alpha / m) * X.T @ (X @ theta - y)
        j_history[i] = cost(X, y, theta)
    return theta, j_history


def normalEqn(X, y):
    theta = np.zeros((X.shape[0], 1))
    theta = np.linalg.inv(X.T @ X) @ X.T @ y  # @等价于.dot()
    return theta


def test1():
    data = np.loadtxt('ex1data1.txt', delimiter=',', dtype=int)
    X = data[:, :1]
    y = data[:, 1].reshape(-1, 1)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    theta = np.zeros((2, 1))
    alpha = 0.01
    iterations = 1600
    theta, j_history = gradientDescent(X, y, theta, alpha, iterations)
    plt.figure()
    plt.plot(X[:, 1], y, '+', label='Training Data')
    plt.plot(X[:, 1], X.dot(theta), label='Liner Regression')
    plt.legend()
    plt.show()
    return theta, j_history


def test2():
    data = np.loadtxt('ex1data2.txt', delimiter=',', dtype=int)
    X = data[:, :2]
    X_Eqn = np.hstack((np.ones((X.shape[0], 1)), X))
    y = data[:, 2].reshape((-1, 1))
    X = (X - np.mean(X, 0)) / np.std(X, 0)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    theta = np.zeros((3, 1))
    alpha = 0.01
    iterations = 400
    theta, j_history = gradientDescent(X, y, theta, alpha, iterations)
    plt.figure()
    plt.plot([i for i in range(iterations)], j_history)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost')
    plt.show()
    theta_normalEqn = normalEqn(X_Eqn, y)
    return theta_normalEqn, theta, j_history


if __name__ == '__main__':
    theta1, j_history1 = test1()
    theta21, theta2, j_history2 = test2()
