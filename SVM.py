import numpy as np
import matplotlib.pyplot as plt

# Assuming the SVM class is defined with fit_SGD, predict methods, etc.

class SVM:
    def __init__(self, C, lr, max_iter):
        self.w = None
        self.b = None
        self.C = C
        self.lr = lr
        self.max_iter = max_iter

    def fit_SGD(self, X, Y):
        # Implement the fitting method with stochastic gradient descent
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.max_iter):
            for i in range(X.shape[0]):
                xi = X[i]
                yi = Y[i]
                condition = yi * (np.dot(self.w, xi) + self.b)
                if condition < 1:
                    self.w -= self.lr * (2 * self.w - self.C * yi * xi)
                    self.b -= self.lr * (-self.C * yi)
                else:
                    self.w -= self.lr * 2 * self.w

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def plot_hyperplane(self, X, Y):
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = (np.dot(xy, self.w) + self.b).reshape(XX.shape)
        plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        plt.show()

# Generate some linearly separable data for demonstration purposes
np.random.seed(1)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20
Y = np.where(np.array(Y) == 0, -1, 1)  # Convert to -1, 1 labels

# Train the SVM model
svm = SVM(C=1, lr=0.01, max_iter=1000)
svm.fit_SGD(X, Y)

# Predict the labels (optional, to verify the performance)
predictions = svm.predict(X)

# Plot the decision boundary
svm.plot_hyperplane(X, Y)
