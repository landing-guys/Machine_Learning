import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, lamda=0):
    first = -sum(y.T @ np.log(sigmoid(X @ theta)) + (1 - y).T @ np.log(1 - sigmoid(X @ theta))) / X.shape[0]
    second = sum(np.power(theta[1:], 2)) * lamda / 2 / X.shape[0]
    grad = (X.T @ (sigmoid(X @ theta) - y) + np.vstack((0, theta[1:])) * lamda) / X.shape[0]
    return first + second, grad


def gradientDescent(X, y, theta, alpha=0.01, num_iters=400, lamda=0):
    j_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        j_history[i], deta_theta = cost(theta, X, y, lamda)
        theta -= alpha * deta_theta
    return theta, j_history


def test1():
    data = np.loadtxt('./ex2data1.txt', delimiter=',')
    X = data[:, :2]
    y = data[:, 2].reshape(-1, 1)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    theta = np.array([-24, 0.2, 0.2]).reshape((-1, 1))
    # theta = np.zeros((3, 1))
    alpha = 0.001
    iterations = 300
    lamda = 0
    theta, j_history = gradientDescent(X, y, theta, alpha, iterations, lamda)
    plotx = [min(X[:, 1]) - 2, max(X[:, 1]) + 2]
    ploty = -(theta[0] + theta[1] * plotx) / theta[2]
    plt.figure()
    plt.plot(data[data[:, 2] == 0][:, 0], data[data[:, 2] == 0][:, 1], '+', label='Not admitted')
    plt.plot(data[data[:, 2] == 1][:, 0], data[data[:, 2] == 1][:, 1], 'o', label='Admitted')
    plt.plot(plotx, ploty, label='Decision Boundary')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    plt.show()
    return theta, j_history


if __name__ == '__main__':
    theta1, j_history1 = test1()
    plt.figure()
    plt.plot([i for i in range(len(j_history1))], j_history1)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
