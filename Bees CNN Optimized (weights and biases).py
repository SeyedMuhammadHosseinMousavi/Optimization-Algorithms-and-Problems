import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

# Preprocess the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Bees Algorithm
class BeesAlgorithm:
    def __init__(self, model, X_train, y_train, n_bees=20, elite_bees=5, patch_size=0.1, iterations=50):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.n_bees = n_bees
        self.elite_bees = elite_bees
        self.patch_size = patch_size
        self.iterations = iterations

        # Initialize bees (weights and biases)
        self.bees = [self.generate_solution() for _ in range(self.n_bees)]

    def generate_solution(self):
        # Flatten weights and biases into a single vector
        weights_and_biases = []
        for layer in self.model.trainable_variables:
            weights_and_biases.append(layer.numpy().flatten())
        return np.concatenate(weights_and_biases)

    def decode_solution(self, solution):
        # Decode the flat vector into weights and biases for the model
        shapes = [layer.shape for layer in self.model.trainable_variables]
        split_points = np.cumsum([np.prod(shape) for shape in shapes])
        decoded = np.split(solution, split_points[:-1])
        decoded = [np.reshape(arr, shape) for arr, shape in zip(decoded, shapes)]
        return decoded

    def set_weights_and_biases(self, solution):
        # Set the model's weights and biases
        decoded = self.decode_solution(solution)
        for layer, new_weights in zip(self.model.trainable_variables, decoded):
            layer.assign(new_weights)

    def fitness(self, solution):
        # Evaluate the model's accuracy on the training data
        self.set_weights_and_biases(solution)
        y_pred = self.model(self.X_train)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(self.y_train, axis=1)), tf.float32)
        ).numpy()
        return accuracy

    def optimize(self):
        for iteration in range(self.iterations):
            # Evaluate fitness for all bees
            fitness_scores = [self.fitness(bee) for bee in self.bees]

            # Sort bees by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            self.bees = [self.bees[i] for i in sorted_indices]

            # Keep elite bees
            elite_bees = self.bees[:self.elite_bees]

            # Scout new bees around elite bees
            for i in range(self.elite_bees, self.n_bees):
                elite_index = i % self.elite_bees
                new_bee = elite_bees[elite_index] + np.random.uniform(
                    -self.patch_size, self.patch_size, size=elite_bees[elite_index].shape
                )
                self.bees[i] = new_bee

            # Reduce patch size over iterations
            self.patch_size *= 0.95

            # Print progress
            best_fitness = fitness_scores[sorted_indices[0]]
            print(f"Iteration {iteration + 1}/{self.iterations}, Best Fitness: {best_fitness:.4f}")

        # Return the best solution
        best_solution = self.bees[0]
        return best_solution


# Define the neural network (simple feedforward model)
def build_model(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    return model


# Build and compile the model
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
model = build_model(input_dim, output_dim)

# Initialize Bees Algorithm
bees_algorithm = BeesAlgorithm(model, X_train, y_train, n_bees=30, elite_bees=5, patch_size=0.1, iterations=50)

# Optimize weights and biases
best_solution = bees_algorithm.optimize()

# Set the optimized weights and biases to the model
bees_algorithm.set_weights_and_biases(best_solution)

# Evaluate the optimized model on the test set
y_test_pred = model(X_test)
y_test_pred_classes = tf.argmax(y_test_pred, axis=1).numpy()
y_test_true_classes = tf.argmax(y_test, axis=1).numpy()

# Print classification report
print("\nClassification Report (Test Data):")
print(classification_report(y_test_true_classes, y_test_pred_classes))
