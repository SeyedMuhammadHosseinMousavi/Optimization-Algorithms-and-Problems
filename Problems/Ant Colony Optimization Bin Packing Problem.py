import numpy as np
import matplotlib.pyplot as plt
import random

# Ant Colony Optimization Parameters
NUM_ANTS = 5
MAX_ITERATIONS = 200
NUM_ITEMS = 78
BIN_CAPACITY = 256
EVAPORATION_RATE = 0.5
PHEROMONE_IMPORTANCE = 1
HEURISTIC_IMPORTANCE = 2
INITIAL_PHEROMONE = 1.0

# Generate a random bin-packing problem
def generate_bin_packing_problem(num_items, bin_capacity):
    weights = np.random.randint(1, bin_capacity // 2, size=num_items)
    return weights, bin_capacity

# Calculate the cost of a solution (total bins used)
def calculate_cost(solution, weights, bin_capacity):
    max_bin_index = max(solution) + 1  # Ensure bins are initialized for all indices
    bins = [0] * max_bin_index  # Initialize bins dynamically

    for item_index, bin_index in enumerate(solution):
        bins[bin_index] += weights[item_index]
        if bins[bin_index] > bin_capacity:
            return float('inf')  # Penalize invalid solutions exceeding bin capacity

    return len([b for b in bins if b > 0])  # Return total number of bins used

# Generate a random initial solution
def generate_initial_solution(weights, bin_capacity):
    solution = np.zeros_like(weights, dtype=int)
    bins = [0]
    for i, weight in enumerate(weights):
        assigned = False
        for bin_index in range(len(bins)):
            if bins[bin_index] + weight <= bin_capacity:
                bins[bin_index] += weight
                solution[i] = bin_index
                assigned = True
                break
        if not assigned:
            bins.append(weight)
            solution[i] = len(bins) - 1
    return solution

# Ant Colony Optimization for Bin Packing Problem
def aco_bin_packing(weights, bin_capacity):
    num_items = len(weights)
    pheromone = np.full((num_items, num_items), INITIAL_PHEROMONE)
    best_solution = None
    best_cost = float('inf')
    iteration_costs = []

    for iteration in range(MAX_ITERATIONS):
        solutions = []
        costs = []

        for ant in range(NUM_ANTS):
            solution = []
            bins = [0]

            for i in range(num_items):
                probabilities = []
                for bin_index in range(len(bins) + 1):
                    if bin_index == len(bins):
                        # New bin
                        if weights[i] <= bin_capacity:
                            probabilities.append((pheromone[i][bin_index - 1] ** PHEROMONE_IMPORTANCE) *
                                                 ((1.0 / (1 + weights[i])) ** HEURISTIC_IMPORTANCE))
                        else:
                            probabilities.append(0)
                    else:
                        if bins[bin_index] + weights[i] <= bin_capacity:
                            probabilities.append((pheromone[i][bin_index] ** PHEROMONE_IMPORTANCE) *
                                                 ((1.0 / (1 + bins[bin_index] + weights[i])) ** HEURISTIC_IMPORTANCE))
                        else:
                            probabilities.append(0)

                probabilities = np.array(probabilities) / sum(probabilities)
                chosen_bin = np.random.choice(range(len(probabilities)), p=probabilities)

                if chosen_bin == len(bins):
                    bins.append(weights[i])
                else:
                    bins[chosen_bin] += weights[i]

                solution.append(chosen_bin)

            cost = calculate_cost(solution, weights, bin_capacity)
            solutions.append(solution)
            costs.append(cost)

            if cost < best_cost:
                best_solution = solution
                best_cost = cost

        # Update pheromones
        pheromone *= (1 - EVAPORATION_RATE)
        for solution, cost in zip(solutions, costs):
            for i, bin_index in enumerate(solution):
                pheromone[i][bin_index] += 1.0 / cost

        iteration_costs.append(best_cost)
        print(f"Iteration {iteration + 1}, Best Cost: {best_cost}")

    return best_solution, best_cost, iteration_costs

def main():
    weights, bin_capacity = generate_bin_packing_problem(NUM_ITEMS, BIN_CAPACITY)
    print("Weights:", weights)
    print("Bin Capacity:", bin_capacity)

    best_solution, best_cost, iteration_costs = aco_bin_packing(weights, bin_capacity)

    print("\nBest Solution:", best_solution)
    print("Number of Bins Used:", best_cost)

    # Plot optimization progress
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_costs, marker='o')
    plt.title("ACO Optimization Progress for Bin Packing")
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost (Number of Bins)")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()