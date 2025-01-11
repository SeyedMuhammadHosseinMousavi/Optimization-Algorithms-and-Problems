import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import distance

# Cuckoo Search Algorithm Parameters
POPULATION_SIZE = 25
MAX_GENERATIONS = 100
NUM_LOCATIONS = 10
MAX_COORDINATE = 100
LEVIY_FLIGHT_STRENGTH = 1.5
DISCOVERY_RATE = 0.25

def create_random_locations(num_locations, max_coordinate):
    return np.random.randint(0, max_coordinate, size=(num_locations, 2))

def calculate_tsp_cost(locations, path):
    cost = 0
    for i in range(len(path)):
        cost += distance.euclidean(locations[path[i]], locations[path[(i + 1) % len(path)]])
    return cost

def levy_flight(Lambda):
    u = np.random.normal(0, 1) * (1 / abs(np.random.normal(0, 1))) ** (1 / Lambda)
    v = np.random.normal(0, 1)
    return u / abs(v) ** (1 / Lambda)

def generate_initial_population(size, num_locations):
    population = []
    for _ in range(size):
        individual = list(range(num_locations))
        random.shuffle(individual)
        population.append(individual)
    return population

def replace_worst_nests(population, fitness, discovery_rate):
    num_replace = int(len(population) * discovery_rate)
    worst_indices = np.argsort(fitness)[-num_replace:]
    for i in worst_indices:
        individual = list(range(len(population[0])))
        random.shuffle(individual)
        population[i] = individual

def cuckoo_search(locations):
    population = generate_initial_population(POPULATION_SIZE, len(locations))
    best_solution = None
    best_cost = float('inf')
    costs_over_time = []

    for generation in range(MAX_GENERATIONS):
        fitness = [calculate_tsp_cost(locations, individual) for individual in population]
        min_cost_index = np.argmin(fitness)
        current_best_cost = fitness[min_cost_index]

        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_solution = population[min_cost_index]

        costs_over_time.append(best_cost)

        for i in range(POPULATION_SIZE):
            cuckoo = population[i][:]
            index1, index2 = random.sample(range(len(cuckoo)), 2)
            cuckoo[index1], cuckoo[index2] = cuckoo[index2], cuckoo[index1]
            cuckoo_cost = calculate_tsp_cost(locations, cuckoo)

            if cuckoo_cost < fitness[i]:
                population[i] = cuckoo

        replace_worst_nests(population, fitness, DISCOVERY_RATE)

        print(f"Generation {generation + 1}, Best TSP Cost: {best_cost}")

    return best_solution, best_cost, costs_over_time

def plot_tsp_solution(locations, solution, title):
    plt.figure(figsize=(8, 6))
    x = [locations[city][0] for city in solution + [solution[0]]]
    y = [locations[city][1] for city in solution + [solution[0]]]
    plt.plot(x, y, marker="o", linestyle="-", color="blue", label="Path")
    plt.scatter(locations[:, 0], locations[:, 1], color="red", s=100, label="Cities")
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    locations = create_random_locations(NUM_LOCATIONS, MAX_COORDINATE)
    best_solution, best_cost, costs_over_time = cuckoo_search(locations)

    print("\nFinal Best TSP Cost:", best_cost)
    print("Best TSP Path:", best_solution)

    plot_tsp_solution(locations, best_solution, "Final TSP Solution")

    # Plot optimization progress
    plt.figure(figsize=(10, 6))
    plt.plot(costs_over_time, marker='o')
    plt.title("Cuckoo Search Optimization of TSP")
    plt.xlabel("Generation")
    plt.ylabel("Best Cost")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()