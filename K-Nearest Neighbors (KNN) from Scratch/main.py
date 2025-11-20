import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from KNN import KNN


def main():
    # Load the Iris dataset from sklearn
    iris = datasets.load_iris()

    # Assign features (X) and labels (y) to variables
    X = iris.data  # Features
    y = iris.target  # Labels

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    # Apply PCA to reduce the feature space from 4 dimensions to 2
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)

    # Plot the transformed training data using the first two principal components
    plt.figure()
    scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap="viridis")
    plt.title("Iris Dataset - PCA (2D projection of 4 features)")
    plt.xlabel("Principal Component 1")  # X-axis for the first principal component
    plt.ylabel("Principal Component 2")  # Y-axis for the second principal component
    plt.legend(handles=scatter.legend_elements()[0], labels=["Setosa", "Versicolor", "Virginica"],)
    plt.show()

    # Initialize and train the KNN model
    knn = KNN(k=3, metric="l2")
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    print(score)


if __name__ == "__main__":
    main()
