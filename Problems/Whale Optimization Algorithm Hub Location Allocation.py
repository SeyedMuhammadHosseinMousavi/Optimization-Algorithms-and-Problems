import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import distance

# Whale Optimization Algorithm Parameters
POPULATION_SIZE = 50
MAX_GENERATIONS = 100
NUM_LOCATIONS = 10
NUM_HUBS = 3
MAX_COORDINATE = 100

def create_random_locations(num_locations, max_coordinate):
    return np.random.randint(0, max_coordinate, size=(num_locations, 2))

def calculate_cost(locations, hubs, allocation):
    total_cost = 0
    for i, hub in enumerate(hubs):
        allocated_points = np.where(allocation == i)[0]
        for point in allocated_points:
            total_cost += distance.euclidean(locations[point], hub)
    return total_cost

def initialize_population(locations, num_hubs, size):
    population = []
    for _ in range(size):
        hubs = locations[np.random.choice(len(locations), num_hubs, replace=False)]
        allocation = np.random.randint(0, num_hubs, size=len(locations))
        population.append((hubs, allocation))
    return population

def update_position_whale(hubs, leader_hubs, a):
    new_hubs = []
    for hub, leader_hub in zip(hubs, leader_hubs):
        r = np.random.rand()
        A = 2 * a * r - a
        C = 2 * r
        D = abs(C * leader_hub - hub)
        new_hub = leader_hub - A * D
        new_hubs.append(new_hub)
    return np.array(new_hubs)

def whale_optimization(locations, num_hubs):
    population = initialize_population(locations, num_hubs, POPULATION_SIZE)
    best_solution = None
    best_cost = float('inf')
    costs_over_time = []

    for generation in range(MAX_GENERATIONS):
        a = 2 - generation * (2 / MAX_GENERATIONS)

        for i in range(POPULATION_SIZE):
            hubs, allocation = population[i]
            cost = calculate_cost(locations, hubs, allocation)
            if cost < best_cost:
                best_solution = (hubs, allocation)
                best_cost = cost

        costs_over_time.append(best_cost)

        leader_hubs, _ = best_solution

        for i in range(POPULATION_SIZE):
            hubs, allocation = population[i]
            new_hubs = update_position_whale(hubs, leader_hubs, a)
            new_allocation = np.random.randint(0, num_hubs, size=len(locations))
            population[i] = (new_hubs, new_allocation)

        print(f"Generation {generation + 1}, Best Cost: {best_cost}")

    return best_solution, best_cost, costs_over_time

def plot_results(locations, hubs, allocation, title):
    plt.figure(figsize=(8, 6))
    colors = ['red', 'green', 'blue', 'purple', 'orange']  # Add more colors if needed

    for i, hub in enumerate(hubs):
        allocated_points = np.where(allocation == i)[0]
        plt.scatter(locations[allocated_points, 0], locations[allocated_points, 1], color=colors[i % len(colors)], label=f"Hub {i + 1} Allocated Points")
        plt.scatter(hub[0], hub[1], color=colors[i % len(colors)], marker="o", s=200, edgecolor="black", label=f"Hub {i + 1}")

        # Draw lines from hub to allocated points
        for point in allocated_points:
            plt.plot([hub[0], locations[point][0]], [hub[1], locations[point][1]], color=colors[i % len(colors)], linestyle='--', linewidth=1)

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    locations = create_random_locations(NUM_LOCATIONS, MAX_COORDINATE)
    best_solution, best_cost, costs_over_time = whale_optimization(locations, NUM_HUBS)

    hubs, allocation = best_solution

    print("\nFinal Best Cost:", best_cost)
    print("Best Hub Locations:", hubs)

    plot_results(locations, hubs, allocation, "Final Hub Location Allocation")

    # Plot optimization progress
    plt.figure(figsize=(10, 6))
    plt.plot(costs_over_time, marker='o')
    plt.title("WOA Optimization of Hub Location Allocation")
    plt.xlabel("Generation")
    plt.ylabel("Best Cost")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()