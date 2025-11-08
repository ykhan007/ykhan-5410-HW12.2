# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Perceptron class
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert labels from 0/1 to -1/1
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_func(linear_output)

    def _unit_step(self, x):
        return np.where(x >= 0, 1, -1)

# Load and prepare the Penguin dataset
df = pd.read_csv("penguins.csv")
df = df.dropna()  # Remove missing data
df = df[df["species"].isin(["Adelie", "Gentoo"])]  # Select two species for binary classification

# Convert species labels to numeric
df["species"] = df["species"].map({"Adelie": 0, "Gentoo": 1})

# Feature columns and target variable
X = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].values
y = df["species"].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the perceptron
p = Perceptron(learning_rate=0.001, n_iters=1000)
p.fit(X_train, y_train)

# Make predictions and evaluate accuracy
predictions = p.predict(X_test)

# Convert back from -1 to 0 if needed
predictions = np.where(predictions == -1, 0, 1)

accuracy = np.mean(predictions == y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Optional: Plot results
plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap='coolwarm')
plt.xlabel("Bill Length (mm)")
plt.ylabel("Bill Depth (mm)")
plt.title("Penguin Classification via Perceptron")
plt.show()
