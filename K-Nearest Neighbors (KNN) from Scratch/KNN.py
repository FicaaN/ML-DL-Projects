class KNN:
    def __init__(self, *, k=3, metric="euclidean", p=3):
        """Initialize the KNN model with parameters
        k: the number of nearest neighbors
        metric: the distance metric to use ('manhattan/l1', 'euclidean/l2', 'minkowski/l3', 'chebyshev/l4')
        p: parameter for Minkowski distance (default is 3 for L3 norm)"""

        self.k = k
        self.metric = metric
        self.p = p

    def distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        # Calculate the distance between two data points (x1, x2)

        # Manhattan Distance (L1 Norm)
        if self.metric in ("manhattan", "l1"):
            return np.sum(np.abs(x1 - x2))
        # Euclidean Distance (L2 Norm)
        elif self.metric in ("euclidean", "l2"):
            return np.sqrt(np.sum((x1 - x2) ** 2))
        # Minkowski Distance (L3 Norm)
        elif self.metric in ("minkowski", "l3"):
            # Use the p value for Minkowski distance
            p = (self.p if hasattr(self, "p") else 3)
            return np.power(np.sum(np.abs(x1 - x2) ** p), 1 / p)
        # Chebyshev Distance (L4 Norm)
        elif self.metric in ("chebyshev", "l4"):
            return np.max(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unsupported distance metric: {self.metric}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model with training data
        X: feature matrix of training data
        y: target labels corresponding to the training data"""

        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be NumPy arrays.")
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> list:
        # Predict the labels for the given test data X
        return [self._predict(x) for x in X]

    def _predict(self, x: np.ndarray) -> int:
        # Helper function for predict(). Predicts the label for a single test point x

        # Calculate the distance from x to all training points
        calculated_distance = [self.distance(x, x_train) for x_train in self.X_train]
        # Find the indices of the k nearest neighbors
        k_indices = np.argsort(calculated_distance)[: self.k]
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Find the most common label (the majority vote)
        most_common = Counter(k_nearest_labels).most_common()
        # Return the label with the highest count
        return most_common[0][0]

    def score(self, X: np.ndarray, y: np.ndarray) -> str:
        # Calculate the accuracy of the model on the test data X and true labels y
        predictions = self.predict(X)
        accuracy = np.mean(np.array(predictions) == np.array(y))
        correctly_labeled = np.sum(np.array(predictions) == np.array(y))
        falsely_labeled = len(y) - correctly_labeled

        # Format and return the results
        result = (
            f"Test size = {len(y)}\n"
            f"Correctly labeled: {correctly_labeled}\n"
            f"Misclassified: {falsely_labeled}\n"
            f"Accuracy: {accuracy:.5f} | {accuracy * 100:.2f}%\n"
        )
        return result
