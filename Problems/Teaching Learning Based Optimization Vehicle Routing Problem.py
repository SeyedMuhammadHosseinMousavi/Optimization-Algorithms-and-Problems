import numpy as np
import matplotlib.pyplot as plt
import time

# Problem Definition
def generate_problem(num_customers, max_demand, area_size):
    customer_locations = np.random.rand(num_customers, 2) * area_size
    depot_location = np.array([area_size / 2, area_size / 2])
    demands = np.random.randint(1, max_demand + 1, size=num_customers)
    locations = np.vstack([depot_location, customer_locations])
    distances = np.linalg.norm(locations[:, np.newaxis] - locations[np.newaxis, :], axis=2)
    return locations, distances, demands, depot_location

# TLBO Algorithm
class TLBO:
    def __init__(self, distances, demands, capacity, population_size, max_generations):
        self.distances = distances
        self.demands = demands
        self.capacity = capacity
        self.population_size = population_size
        self.max_generations = max_generations
        self.num_customers = len(demands)
        self.num_locations = len(distances)
        self.population = [self.generate_solution() for _ in range(population_size)]

    def generate_solution(self):
        solution = np.random.permutation(np.arange(1, self.num_customers + 1))
        return solution

    def fitness(self, solution):
        routes = self.decode_routes(solution)
        total_distance = 0
        for route in routes:
            route_distance = 0
            route = [0] + route + [0]
            for i in range(len(route) - 1):
                route_distance += self.distances[route[i], route[i + 1]]
            total_distance += route_distance
        return total_distance

    def decode_routes(self, solution):
        routes = []
        current_route = []
        current_demand = 0
        for customer in solution:
            if current_demand + self.demands[customer - 1] > self.capacity:
                routes.append(current_route)
                current_route = []
                current_demand = 0
            current_route.append(customer)
            current_demand += self.demands[customer - 1]
        if current_route:
            routes.append(current_route)
        return routes

    def teacher_phase(self, population):
        best_solution = min(population, key=self.fitness)
        best_fitness = self.fitness(best_solution)
        new_population = []
        for solution in population:
            r = np.random.rand()
            new_solution = solution + r * (best_solution - solution)
            new_solution = np.clip(new_solution, 1, self.num_customers).astype(int)
            new_population.append(new_solution)
        return new_population

    def learner_phase(self, population):
        new_population = []
        for i, solution in enumerate(population):
            partner_idx = np.random.choice([j for j in range(len(population)) if j != i])
            partner = population[partner_idx]
            r = np.random.rand()
            if self.fitness(partner) < self.fitness(solution):
                new_solution = solution + r * (partner - solution)
            else:
                new_solution = solution - r * (partner - solution)
            new_solution = np.clip(new_solution, 1, self.num_customers).astype(int)
            new_population.append(new_solution)
        return new_population

    def optimize(self):
        for generation in range(self.max_generations):
            self.population = self.teacher_phase(self.population)
            self.population = self.learner_phase(self.population)
            best_solution = min(self.population, key=self.fitness)
            best_routes = self.decode_routes(best_solution)
            print(f"Generation {generation + 1}: Best Distance = {self.fitness(best_solution):.2f}")
            self.plot_progress(best_routes, f"Generation {generation + 1}: Distance = {self.fitness(best_solution):.2f}")
            time.sleep(0.5)  # Simulate real-time processing

        best_solution = min(self.population, key=self.fitness)
        best_routes = self.decode_routes(best_solution)
        return best_solution, best_routes, self.fitness(best_solution)

    def plot_progress(self, routes, title):
        plt.clf()
        plt.scatter(locations[1:, 0], locations[1:, 1], c='blue', label='Customers')
        plt.scatter(locations[0, 0], locations[0, 1], c='red', label='Depot', marker='x')

        for route in routes:
            route = [0] + route + [0]
            plt.plot(locations[route, 0], locations[route, 1], '-o')

        plt.title(title)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid()
        plt.pause(0.1)  # Update plot in real-time

# Visualization
plt.ion()  # Interactive mode for real-time updates

def plot_routes(routes, locations, title="Vehicle Routing Problem Solution"):
    plt.figure(figsize=(10, 6))
    plt.scatter(locations[1:, 0], locations[1:, 1], c='blue', label='Customers')
    plt.scatter(locations[0, 0], locations[0, 1], c='red', label='Depot', marker='x')

    for route in routes:
        route = [0] + route + [0]
        plt.plot(locations[route, 0], locations[route, 1], '-o')

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Generate a new problem instance each time
    num_customers = 15
    max_demand = 10
    area_size = 100
    vehicle_capacity = 25
    population_size = 20
    max_generations = 30

    locations, distances, demands, depot_location = generate_problem(num_customers, max_demand, area_size)

    print("Customer Locations:", locations[1:])
    print("Depot Location:", depot_location)
    print("Demands:", demands)

    # Solve the problem using TLBO
    tlbo = TLBO(distances, demands, vehicle_capacity, population_size, max_generations)
    best_solution, best_routes, best_distance = tlbo.optimize()

    # Print Results
    print("Best Solution (Encoded):", best_solution)
    print("Best Routes:", best_routes)
    print("Best Distance:", best_distance)

    # Final Visualization
    plt.ioff()  # Turn off interactive mode
    plot_routes(best_routes, locations, title=f"Final Solution: Distance = {best_distance:.2f}")
