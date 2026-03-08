import numpy as np

def one_hot(y, num_classes=10):
    o = np.zeros((y.size, num_classes))
    o[np.arange(y.size), y] = 1
    return o

class ReLU:
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)
    def backward(self, dA):
        return dA * (self.Z > 0)


class Softmax:
    def forward(self, Z):
        Z = Z - np.max(Z, axis=1, keepdims=True)
        exp = np.exp(Z)
        self.out = exp / np.sum(exp, axis=1, keepdims=True)
        return self.out
    def backward(self, dA):
        return dA


class MSE:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)
    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]


class CrossEntropy:
    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(y_pred, 1e-9, 1-1e-9)
        self.y_true = y_true
        return -np.mean(np.sum(y_true * np.log(self.y_pred), axis=1))
    def backward(self):
        return (self.y_pred - self.y_true) / self.y_true.shape[0]


def resolve_activation(name):
    if name == "relu":
        return ReLU()
    if name == "softmax":
        return Softmax()
    return None


def resolve_loss(name):
    if name == "mse":
        return MSE()
    if name in ["crossentropy", "ce", "categorical_crossentropy"]:
        return CrossEntropy()
    return MSE()


class Layer:
    def __init__(self, units, input_shape=None, activation=None):
        self.units = units
        self.input_shape = input_shape
        self.activation = resolve_activation(activation)

    def build(self, input_dim):
        self.W = np.random.randn(input_dim, self.units) * np.sqrt(2 / input_dim)
        self.b = np.zeros(self.units)

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b
        if self.activation:
            return self.activation.forward(self.Z)
        return self.Z

    def backward(self, dA):
        if self.activation:
            dZ = self.activation.backward(dA)
        else:
            dZ = dA
        self.dW = self.X.T @ dZ / self.X.shape[0]
        self.db = dZ.mean(axis=0)
        return dZ @ self.W.T


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def update(self, param, grad, key):
        if key not in self.m:
            self.m[key] = np.zeros_like(grad)
            self.v[key] = np.zeros_like(grad)

        self.t += 1
        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)

        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class Automation:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def run(self, optimizer="adam", loss="mse", metrics=None):
        self.loss = resolve_loss(loss)
        self.metrics = metrics or []
        self.optimizer = Adam()

        for i, layer in enumerate(self.layers):
            if layer.input_shape:
                input_dim = layer.input_shape[0]
            else:
                input_dim = self.layers[i-1].units
            layer.build(input_dim)

    def fit(self, X, y, epochs=10, batch_size=32, verbose=1):
        n = X.shape[0]

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X, y = X[idx], y[idx]

            for i in range(0, n, batch_size):
                xb = X[i:i+batch_size]
                yb = y[i:i+batch_size]

                out = xb
                for layer in self.layers:
                    out = layer.forward(out)

                loss = self.loss.forward(out, yb)

                grad = self.loss.backward()
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

                for li, layer in enumerate(self.layers):
                    layer.W = self.optimizer.update(layer.W, layer.dW, f"W{li}")
                    layer.b = self.optimizer.update(layer.b, layer.db, f"b{li}")

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.4f}")

    def predict(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def results(self, X, y):
        pred = self.predict(X)
        mse = np.mean((pred - y) ** 2)
        mae = np.mean(np.abs(pred - y))
        return mse, mae
