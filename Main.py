import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as qnp


class QuantumPolynomialRegression:
    def __init__(self, n_qbit=6, n_layers=4):
        self.n_qbit = n_qbit
        self.n_layers = n_layers
        self.dev = qml.device("lightning.qubit", wires=n_qbit)
        self.params = qnp.random.normal(0, 0.05, (n_layers, n_qbit, 3), requires_grad=True)
        self.qnode = qml.QNode(self.quantum_circuit, self.dev, diff_method="adjoint")

    def quantum_circuit(self, x, params):
        # Data encoding
        for i in range(self.n_qbit):
            qml.RY(x * np.pi, i)
            if i < self.n_qbit - 1:
                qml.RZ(x * np.pi / 2, i)

        for i in range(self.n_qbit):
            qml.Hadamard(i)

        # Variational layers
        for layer in range(self.n_layers):
            for wire in range(self.n_qbit):
                qml.RX(params[layer, wire, 0], wire)
                qml.RY(params[layer, wire, 1], wire)
                qml.RZ(params[layer, wire, 2], wire)

            # Entangling
            for wire in range(0, self.n_qbit - 1, 2):
                qml.CNOT([wire, wire + 1])
            for wire in range(1, self.n_qbit - 1, 2):
                qml.CNOT([wire, wire + 1])

            if self.n_qbit > 2:
                qml.CNOT([self.n_qbit - 1, 0])

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    def predict(self, X):
        return qnp.array([self.qnode(x, self.params) for x in X])

    def cost_func(self, params, X, y):
        predictions = qnp.array([self.qnode(x, params) for x in X])
        return qnp.mean((predictions - qnp.array(y)) ** 2)

    def fit(self, X, y, epochs=200):
        # Normalize data
        self.input_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(-0.9, 0.9))

        X_norm = self.input_scaler.fit_transform(X.reshape(-1, 1)).flatten()
        y_norm = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)

        optimizer = qml.AdamOptimizer(stepsize=0.08)

        print("Training...")
        for epoch in range(epochs):
            self.params = optimizer.step(lambda p: self.cost_func(p, X_train, y_train), self.params)

            if epoch % 50 == 0:
                loss = self.cost_func(self.params, X_train, y_train)
                print(f"Epoch {epoch}: Loss = {float(loss):.4f}")

        # Evaluation
        y_pred_norm = self.predict(X_test)
        y_pred = self.target_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
        y_test_orig = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        mse = mean_squared_error(y_test_orig, y_pred)
        r2 = r2_score(y_test_orig, y_pred)

        print(f"MSE: {mse:.4f}, R²: {r2:.4f}")
        return self

    def predict_new(self, X):
        X_norm = self.input_scaler.transform(X.reshape(-1, 1)).flatten()
        y_pred_norm = self.predict(X_norm)
        return self.target_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

    def plot_results(self, X, y, save_path="quantum_regression.png"):
        y_pred = self.predict_new(X)
        sorted_idx = np.argsort(X)

        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, alpha=0.6, s=30, label='Data', color='blue')
        plt.plot(X[sorted_idx], y_pred[sorted_idx], 'r-', linewidth=2, label='Predictions')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Quantum Polynomial Regression')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        plt.show()


# Usage
if __name__ == "__main__":
    np.random.seed(42)
    X = np.linspace(-2, 2, 120)
    y_true = 0.5 * X ** 3 + 2 * X ** 2 + X + 1
    y = y_true + np.random.normal(0, 0.5, len(X))

    model = QuantumPolynomialRegression(n_qbit=6, n_layers=4)
    model.fit(X, y)
    model.plot_results(X, y)

