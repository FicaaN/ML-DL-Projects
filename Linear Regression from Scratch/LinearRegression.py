import numpy as np


class LinearRegression:
    def __init__(self, *, lr=0.01, epochs=1000):
        """
        Initializes the Linear Regression model (using Gradient Descent)
        - lr (float): Learning rate for gradient descent, defaults to 0.01
        - epochs (int): Number of iterations for training, defaults to 1000
        """

        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model using gradient descent
        - X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        - y (np.ndarray): Target values of shape (n_samples,)
        """

        n_samples, n_features = X.shape
        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            # Compute predictions
            y_pred = np.dot(X, self.weights) + self.bias
            # Compute gradient for weights
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            # Compute gradient for bias
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained model
        - X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        """

        if self.weights is None or self.bias is None:
            raise ValueError("Model is not trainder yet. Call 'fit' before 'predict'.")

        # Compute predictions
        return np.dot(X, self.weights) + self.bias

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Computes Mean Squared Error (MSE), Mean Absolute Error (MAE),
        and R-squared (R²) Score
        - y_true (np.ndarray): Actual target values.
        - y_pred (np.ndarray): Predicted target values.
        """
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
        ss_residual = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
        r2 = 1 - (ss_residual / ss_total)

        result = (
            f"Mean Squared Error (MSE): {mse:.3f}\n"
            f"Mean Absolute Error (MAE): {mae:.3f}\n"
            f"R-squared (R²) Score: {r2:.3f}"
        )

        return result
