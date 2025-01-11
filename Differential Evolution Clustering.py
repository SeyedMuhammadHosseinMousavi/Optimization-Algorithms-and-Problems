import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin
import matplotlib.pyplot as plt

# Step 1: Load and Prepare the Iris Dataset
def load_and_preprocess_data():
    iris = load_iris()
    X = iris.data  # Features
    y = iris.target  # Labels (not used in clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Step 2: Define Differential Evolution (DE)
class DE:
    def __init__(self, n_clusters, n_population, n_iterations, X):
        self.n_clusters = n_clusters
        self.n_population = n_population
        self.n_iterations = n_iterations
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialize population (random cluster centers)
        self.population = np.random.rand(n_population, n_clusters, self.n_features)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.cost_history = []

    def fitness(self, cluster_centers):
        # Assign points to nearest cluster center
        labels = pairwise_distances_argmin(self.X, cluster_centers)
        # Compute intra-cluster distance (sum of squared distances)
        score = sum(np.sum((self.X[labels == i] - center) ** 2)
                    for i, center in enumerate(cluster_centers))
        return score

    def optimize(self):
        F = 0.8  # Scaling factor
        CR = 0.9  # Crossover probability
        for iteration in range(self.n_iterations):
            new_population = np.copy(self.population)
            for i in range(self.n_population):
                # Mutation: Select three random individuals different from i
                indices = [idx for idx in range(self.n_population) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = a + F * (b - c)

                # Crossover: Combine mutant vector and target vector
                crossover_mask = np.random.rand(*mutant_vector.shape) < CR
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])

                # Selection: Evaluate and select the better individual
                trial_score = self.fitness(trial_vector)
                target_score = self.fitness(self.population[i])
                if trial_score < target_score:
                    new_population[i] = trial_vector
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial_vector

            self.population = new_population
            self.cost_history.append(self.global_best_score)
            print(f"Iteration {iteration + 1}/{self.n_iterations}, Best Score: {self.global_best_score}")

        return self.global_best_position, self.cost_history

# Step 3: Clustering with DE-generated Centers
def clustering_with_de(X, n_clusters, n_population, n_iterations):
    de = DE(n_clusters, n_population, n_iterations, X)
    best_centers, cost_history = de.optimize()
    labels = pairwise_distances_argmin(X, best_centers)
    return labels, best_centers, cost_history

# Step 4: Evaluate the Clustering
def evaluate_clustering(X, labels, centers):
    quantization_error = sum(np.sum((X[labels == i] - center) ** 2)
                             for i, center in enumerate(centers))
    intra_cluster_distances = [np.sum((X[labels == i] - center) ** 2) 
                               for i, center in enumerate(centers)]
    inter_cluster_distances = np.min(
        [np.linalg.norm(center1 - center2) 
         for i, center1 in enumerate(centers) 
         for j, center2 in enumerate(centers) if i != j])
    print(f"Quantization Error: {quantization_error:.4f}")
    print(f"Intra-cluster Distances: {intra_cluster_distances}")
    print(f"Inter-cluster Distance: {inter_cluster_distances:.4f}")
    return quantization_error, intra_cluster_distances, inter_cluster_distances

# Step 5: Visualize the Clustering Result
def visualize_results(X, labels, centers, cost_history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Clustering result
    axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.7)
    axes[0].scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centers')
    axes[0].set_title("Clustering Result with DE")
    axes[0].legend()

    # DE iteration cost
    axes[1].plot(range(1, len(cost_history) + 1), cost_history, marker='o')
    axes[1].set_title("DE Iteration Cost")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Cost (Fitness)")

    plt.tight_layout()
    plt.show()

# Step 6: Main Function
def main():
    X, y = load_and_preprocess_data()
    n_clusters = 3
    n_population = 10
    n_iterations = 100

    labels, centers, cost_history = clustering_with_de(X, n_clusters, n_population, n_iterations)
    evaluate_clustering(X, labels, centers)
    visualize_results(X, labels, centers, cost_history)

if __name__ == "__main__":
    main()
