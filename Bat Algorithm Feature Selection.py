import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Bat Algorithm Parameters
POPULATION_SIZE = 3
MAX_GENERATIONS = 30
LOUDNESS = 0.5
PULSE_RATE = 0.5
FREQ_MIN = 0
FREQ_MAX = 2
NUM_SAMPLES = 500
NUM_FEATURES = 10
NUM_CLASSES = 3
NUM_SELECTED_FEATURES = 5

# Generate random dataset
def generate_random_data(num_samples, num_features, num_classes):
    X = np.random.rand(num_samples, num_features)
    y = np.random.randint(0, num_classes, num_samples)
    return X, y

# Cost function
def cost_function(X, y, selected_features):
    if np.sum(selected_features) == 0:
        return float('inf')

    X_selected = X[:, selected_features == 1]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return 1 - accuracy_score(y_test, y_pred)  # Minimize error

# Initialize population
def initialize_population(size, num_features):
    population = np.zeros((size, num_features))
    for i in range(size):
        selected_indices = np.random.choice(num_features, NUM_SELECTED_FEATURES, replace=False)
        population[i, selected_indices] = 1
    return population

# Update position using Bat Algorithm
def update_position(individual, velocity, frequency, best_individual):
    velocity += (individual - best_individual) * frequency
    new_position = individual + velocity
    return np.clip(np.round(new_position), 0, 1)

# Bat Algorithm for Feature Selection
def bat_algorithm(X, y):
    num_features = X.shape[1]
    population = initialize_population(POPULATION_SIZE, num_features)
    velocities = np.zeros((POPULATION_SIZE, num_features))
    cost = np.array([cost_function(X, y, individual) for individual in population])

    best_individual = population[np.argmin(cost)]
    best_cost = np.min(cost)
    costs_over_time = []

    for generation in range(MAX_GENERATIONS):
        for i in range(POPULATION_SIZE):
            frequency = FREQ_MIN + (FREQ_MAX - FREQ_MIN) * random.random()
            velocities[i] = update_position(population[i], velocities[i], frequency, best_individual)
            candidate = update_position(population[i], velocities[i], frequency, best_individual)

            if random.random() > PULSE_RATE:
                candidate = best_individual.copy()
                mutation_index = random.randint(0, num_features - 1)
                candidate[mutation_index] = 1 - candidate[mutation_index]

            candidate_cost = cost_function(X, y, candidate)
            if candidate_cost < cost[i] and random.random() < LOUDNESS:
                population[i] = candidate
                cost[i] = candidate_cost

                if candidate_cost < best_cost:
                    best_individual = candidate
                    best_cost = candidate_cost

        costs_over_time.append(best_cost)
        print(f"Generation {generation + 1}, Best Cost: {best_cost:.4f}")

    return best_individual, 1 - best_cost, costs_over_time

def main():
    X, y = generate_random_data(NUM_SAMPLES, NUM_FEATURES, NUM_CLASSES)
    # Split data for original accuracy evaluation
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.3, random_state=42)
    model_full = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model_full.fit(X_train_full, y_train_full)
    y_pred_full = model_full.predict(X_test_full)
    original_accuracy = accuracy_score(y_test_full, y_pred_full)

    best_features, best_accuracy, costs_over_time = bat_algorithm(X, y)

    # Evaluate accuracy with selected features
    X_selected = X[:, best_features == 1]
    X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    model_selected = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model_selected.fit(X_train_selected, y_train_selected)
    y_pred_selected = model_selected.predict(X_test_selected)
    selected_accuracy = accuracy_score(y_test_selected, y_pred_selected)

    print("\nOriginal Accuracy with All Features:", original_accuracy)
    print("Selected Features (1=selected, 0=not selected):", best_features)
    print("Accuracy with Selected Features:", selected_accuracy)

    # Count selected features
    print("Number of Selected Features:", np.sum(best_features))

    # Plot optimization progress
    plt.figure(figsize=(10, 6))
    plt.plot(costs_over_time, marker='o')
    plt.title("Bat Algorithm Optimization Progress")
    plt.xlabel("Generation")
    plt.ylabel("Best Cost")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()