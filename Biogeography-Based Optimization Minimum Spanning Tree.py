import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

# BBO Algorithm Parameters
POPULATION_SIZE = 20
MUTATION_RATE = 0.2
MAX_GENERATIONS = 500
NUM_NODES = 8
MAX_EDGE_WEIGHT = 20

# Create a random graph
def create_random_graph(num_nodes, max_edge_weight):
    graph = nx.complete_graph(num_nodes)
    for (u, v) in graph.edges():
        graph.edges[u, v]['weight'] = random.randint(1, max_edge_weight)
    return graph

# Fitness function for MST (lower cost is better)
def fitness_function(graph, individual):
    mst_cost = sum(graph.edges[edge]['weight'] for edge in individual)
    return 1 / mst_cost  # Higher fitness for lower cost

# Generate initial population
def generate_population(graph, size):
    population = []
    for _ in range(size):
        edges = list(graph.edges)
        random.shuffle(edges)
        population.append(edges[:NUM_NODES - 1])  # Ensure a spanning tree
    return population

# Selection function (roulette wheel)
def select(population, fitnesses):
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    selected_index = np.random.choice(len(population), p=probabilities)
    return population[selected_index]

# Mutation function
def mutate(individual, graph):
    if random.random() < MUTATION_RATE:
        new_edge = random.choice(list(graph.edges))
        individual[random.randint(0, len(individual) - 1)] = new_edge
    return individual

# Plot final MST
def plot_final_mst(graph, edges):
    mst_graph = nx.Graph()
    mst_graph.add_edges_from(edges)
    pos = nx.spring_layout(graph)

    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    nx.draw(mst_graph, pos, with_labels=True, edge_color='red', width=2)
    plt.title("Final MST")
    plt.show()

# BBO Main Function
def bbo_mst(graph):
    population = generate_population(graph, POPULATION_SIZE)
    best_solution = None
    best_cost = float('inf')
    iteration_costs = []

    for generation in range(MAX_GENERATIONS):
        fitnesses = [fitness_function(graph, individual) for individual in population]
        best_index = np.argmax(fitnesses)
        current_best_cost = 1 / fitnesses[best_index]

        if current_best_cost < best_cost:
            best_solution = population[best_index]
            best_cost = current_best_cost

        iteration_costs.append(best_cost)

        new_population = []
        for _ in range(POPULATION_SIZE):
            parent = select(population, fitnesses)
            offspring = mutate(parent.copy(), graph)
            new_population.append(offspring)

        population = new_population

        print(f"Generation {generation + 1}, Best MST Cost: {best_cost}")

    return best_solution, best_cost, iteration_costs

# Main Execution
if __name__ == "__main__":
    random_graph = create_random_graph(NUM_NODES, MAX_EDGE_WEIGHT)
    best_mst, best_mst_cost, costs_over_time = bbo_mst(random_graph)

    print("\nFinal Best MST Cost:", best_mst_cost)
    print("Best MST Edges:", best_mst)

    # Plot the final MST
    plot_final_mst(random_graph, best_mst)

    # Plot iterations over time
    plt.figure(figsize=(10, 6))
    plt.plot(costs_over_time, marker='o')
    plt.title("BBO Optimization of MST")
    plt.xlabel("Generation")
    plt.ylabel("Best MST Cost")
    plt.grid()
    plt.show()
