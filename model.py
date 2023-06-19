import numpy as np

class LinearModel:

    def __init__(self, X, y, alpha, num_iters=1000):
        self.cf = self.intercept = None
        self.alpha = alpha
        self.num_iters = num_iters
        self.num_samples = len(y)
        self.num_featues = np.size(X, 1)
        self.X = np.hstack(np.ones(self.num_samples, 1), (X - np.mean(X, 0)) / (X - np.std(X, 0)))
        self.y = y[:, np.newaxis]
        self.weights = np.zeros((self.num_samples+1, 1))

    def Fit(self, X, y):
        for i in range(self.num_iters):
            self.weights = self.weights - (self.alpha/self.num_samples) * \
            self.X.T @ (self.X @ self.weights - self.y)
        self.intercept = self.weights[0]
        self.cf = self.weights[1:]
        return self

    def Predict(self, X):
        num_samples = np.size(X, 0)
        y = np.hstack(np.ones(num_samples, 1), (X - np.mean(X, 0) /
                                                (X - np.std(X, 0)))) @ self.weights
        return y

    def Score(self, X, y):
        num_samples = np.size(X, 0)
        X = np.hstack(np.ones(num_samples, 1), (X - np.mean(X, 0) /
                                                (X - np.std(X, 0))))
        y = y[:, np.newaxis]
        predictions = self.weights @ X
        return 1 - (((y-predictions)**2).sum() / ((y-y.mean)**2).sum())

    def GetWeights(self):
        return self.weights
