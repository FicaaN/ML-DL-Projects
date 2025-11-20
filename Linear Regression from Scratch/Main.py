import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression


def main():
    # Generate a synthetic dataset
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression(lr=0.01, epochs=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(model.evaluate(y_test, y_pred))

    plt.scatter(X_test, y_test, color="blue", label="Actual data")
    plt.plot(X_test, y_pred, color="red", linewidth=1.5, label="Regression line")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Linear Regression Model Fit")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
