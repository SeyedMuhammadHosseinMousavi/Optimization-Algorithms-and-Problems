import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_friedman1

# Load a sample regression dataset (Friedman #1 dataset)
X, y = make_friedman1(n_samples=200, n_features=5, noise=0.1, random_state=42)

# Use only the first feature for simplicity (can adjust as needed)
X = X[:, :1]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Harmony Search algorithm
class HarmonySearch:
    def __init__(self, obj_func, bounds, hms=20, hmcr=0.9, par=0.3, max_iter=200):
        self.obj_func = obj_func
        self.bounds = bounds
        self.hms = hms
        self.hmcr = hmcr
        self.par = par
        self.max_iter = max_iter
        self.harmony_memory = []

    def initialize(self):
        for _ in range(self.hms):
            harmony = [np.random.uniform(low, high) for low, high in self.bounds]
            self.harmony_memory.append(harmony)

    def improvise(self):
        new_harmony = []
        for i, (low, high) in enumerate(self.bounds):
            if np.random.rand() < self.hmcr:
                new_value = np.random.choice([h[i] for h in self.harmony_memory])
                if np.random.rand() < self.par:
                    new_value += np.random.uniform(-1, 1) * (high - low) * 0.01
            else:
                new_value = np.random.uniform(low, high)
            new_harmony.append(np.clip(new_value, low, high))
        return new_harmony

    def optimize(self):
        self.initialize()
        for _ in range(self.max_iter):
            new_harmony = self.improvise()
            new_score = self.obj_func(new_harmony)
            worst_idx = np.argmax([self.obj_func(h) for h in self.harmony_memory])
            if new_score < self.obj_func(self.harmony_memory[worst_idx]):
                self.harmony_memory[worst_idx] = new_harmony
        best_idx = np.argmin([self.obj_func(h) for h in self.harmony_memory])
        return self.harmony_memory[best_idx]

# Objective function for regression (minimize MSE)
def objective(params):
    degree = int(params[0])
    coeffs = params[1:degree + 2]  # Adjust number of coefficients to match degree + intercept
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)  # Add intercept term
    y_pred = np.dot(X_poly_train, np.array(coeffs))  # Predict with coefficients
    return mean_squared_error(y_train, y_pred)

# Set bounds for Harmony Search
max_poly_degree = 5  # Maximum degree of the polynomial
bounds = [(1, max_poly_degree)] + [(-10, 10) for _ in range(max_poly_degree + 1)]  # +1 for intercept

# Run Harmony Search
hs = HarmonySearch(obj_func=objective, bounds=bounds, max_iter=200)
best_params = hs.optimize()

# Extract the best degree and coefficients
best_degree = int(best_params[0])
best_coeffs = best_params[1:best_degree + 2]  # Include intercept term

# Use the best polynomial degree and coefficients for plotting
poly = PolynomialFeatures(degree=best_degree)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
y_train_pred = np.dot(X_poly_train, np.array(best_coeffs))
y_test_pred = np.dot(X_poly_test, np.array(best_coeffs))

# Calculate Mean Squared Error (MSE)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"MSE (Train): {mse_train:.4f}")
print(f"MSE (Test): {mse_test:.4f}")
print(f"Optimized Polynomial Degree: {best_degree}")

# Plot the regression curve
X_range = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_range_pred = np.dot(X_range_poly, np.array(best_coeffs))

plt.scatter(X, y, color='blue', label='Data Samples')
plt.plot(X_range, y_range_pred, color='red', linewidth=2, label=f'Degree {best_degree} Fit')
plt.title('Nonlinear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
