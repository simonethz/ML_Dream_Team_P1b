import numpy as np
import pandas as pd

def transform_features(X):
    X_transformed = np.zeros((700, 21))
    X_transformed[:, 0:5] = X
    X_transformed[:, 5:10] = X**2
    X_transformed[:, 10:15] = np.exp(X)
    X_transformed[:, 15:20] = np.cos(X)
    X_transformed[:, 20] = np.ones(np.size(X, axis=0))
    assert X_transformed.shape == (700, 21)
    return X_transformed

def logistic_gradient(w, X, y):
    n = X.shape[0]
    scores = y * (X @ w)
    factor = -y / (1 + np.exp(scores))
    grad = (X.T @ factor) / n
    return grad

def fit_logistic_regression(X, y):
    y = 2*y - 1
    weights = np.zeros((21,))
    X_transformed = transform_features(X)
<<<<<<< HEAD

    model = LogisticRegression(fit_intercept=False, max_iter=10000, C=10)
    model.fit(X_transformed, y)
    weights = model.coef_.reshape(-1)

    # inverse_X2 = np.linalg.inv(X_transformed.T @ X_transformed)
    # weights = inverse_X2 @ X_transformed.T @ y
=======
    lamda = 1
    grad = logistic_gradient(weights, X_transformed, y)
    while np.linalg.norm(grad)>5e-5:
        weights = weights - grad * lamda
        grad = logistic_gradient(weights, X_transformed, y)
>>>>>>> d91a94c (final version)
    assert weights.shape == (21,)
    weights = weights / np.linalg.norm(weights)
    return weights

if __name__ == "__main__":
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    print(data.head())
    X = data.to_numpy()
    w = fit_logistic_regression(X, y)
    np.savetxt("./results.csv", w, fmt="%.12f")
