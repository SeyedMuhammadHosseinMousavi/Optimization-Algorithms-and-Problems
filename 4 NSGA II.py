import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import time
import random

# Define all 20 benchmark functions
def ackley(x):
    x = np.array(x)
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.exp(1)

def booth(x):
    x = np.array(x)
    return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

def rastrigin(x):
    x = np.array(x)
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock(x):
    x = np.array(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def schwefel(x):
    x = np.array(x)
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def sphere(x):
    x = np.array(x)
    return np.sum(x**2)

def michalewicz(x):
    x = np.array(x)
    m = 10
    return -np.sum(np.sin(x) * np.sin(((np.arange(len(x)) + 1) * x**2) / np.pi)**(2 * m))

def zakharov(x):
    x = np.array(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * (np.arange(len(x)) + 1) * x)
    return sum1 + sum2**2 + sum2**4

def eggholder(x):
    x = np.array(x)
    return -(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1] + 47)))) - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47))))

def beale(x):
    x = np.array(x)
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

def trid(x):
    x = np.array(x)
    return np.sum((x - 1)**2) - np.sum(x[:-1] * x[1:])

def dixon_price(x):
    x = np.array(x)
    return (x[0] - 1)**2 + np.sum([(i + 1) * (2 * x[i]**2 - x[i-1])**2 for i in range(1, len(x))])

def cross_in_tray(x):
    x = np.array(x)
    fact1 = np.sin(x[0]) * np.sin(x[1])
    fact2 = np.exp(abs(100 - np.sqrt(x[0]**2 + x[1]**2) / np.pi))
    return -0.0001 * (abs(fact1 * fact2) + 1)**0.1

def griewank(x):
    x = np.array(x)
    return 1 + np.sum(x**2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

def levy(x):
    x = np.array(x)
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def matyas(x):
    x = np.array(x)
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

def goldstein_price(x):
    x = np.array(x)
    term1 = 1 + ((x[0] + x[1] + 1)**2) * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
    term2 = 30 + ((2*x[0] - 3*x[1])**2) * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2)
    return term1 * term2

def powell(x):
    x = np.array(x)
    term1 = (x[0] + 10*x[1])**2
    term2 = 5 * (x[2] - x[3])**2
    term3 = (x[1] - 2*x[2])**4
    term4 = 10 * (x[0] - x[3])**4
    return term1 + term2 + term3 + term4

def bird(x):
    x = np.array(x)
    return np.sin(x[0]) * np.exp((1 - np.cos(x[1]))**2) + np.cos(x[1]) * np.exp((1 - np.sin(x[0]))**2) + (x[0] - x[1])**2

def pyramid(x):
    x = np.array(x)
    return np.sum(np.abs(x))

# NSGA-II Implementation
def nsga2(func, bounds, population_size=50, generations=100, mutation_rate=0.1, crossover_rate=0.9):
    dimensions = len(bounds)

    # Initialize population
    population = [np.array([random.uniform(b[0], b[1]) for b in bounds]) for _ in range(population_size)]

    # Multi-objective transformation (original function + auxiliary objective)
    def multiobjective_func(x):
        return func(x), np.sum(np.abs(x))

    fitness = [multiobjective_func(ind) for ind in population]

    # Helper functions for NSGA-II
    def dominates(f1, f2):
        return all(f1_i <= f2_i for f1_i, f2_i in zip(f1, f2)) and any(f1_i < f2_i for f1_i, f2_i in zip(f1, f2))

    def non_dominated_sort(fitness):
        fronts = [[]]
        domination_count = np.zeros(len(fitness))
        dominated_solutions = [[] for _ in range(len(fitness))]
        rank = np.zeros(len(fitness))

        for i in range(len(fitness)):
            for j in range(len(fitness)):
                if dominates(fitness[i], fitness[j]):
                    dominated_solutions[i].append(j)
                elif dominates(fitness[j], fitness[i]):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                rank[i] = 0
                fronts[0].append(i)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        fronts.pop()  # Remove the last empty front
        return fronts

    def crowding_distance(front, fitness):
        distances = np.zeros(len(front))
        num_objectives = len(fitness[0])

        for m in range(num_objectives):
            sorted_indices = np.argsort([fitness[i][m] for i in front])
            sorted_fitness = [fitness[i][m] for i in sorted_indices]
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = float('inf')
            for i in range(1, len(front) - 1):
                distances[sorted_indices[i]] += (sorted_fitness[i + 1] - sorted_fitness[i - 1]) / (
                        sorted_fitness[-1] - sorted_fitness[0])
        return distances

    def mutate(individual):
        for i in range(dimensions):
            if random.random() < mutation_rate:
                individual[i] = random.uniform(bounds[i][0], bounds[i][1])
        return individual

    def crossover(parent1, parent2):
        if random.random() < crossover_rate:
            point = random.randint(1, dimensions - 1)
            return np.concatenate((parent1[:point], parent2[point:]))
        return parent1

    # NSGA-II Main Loop
    for generation in range(generations):
        offspring = []
        for _ in range(population_size // 2):
            parent1, parent2 = random.sample(population, 2)
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            offspring.append(child1)
            offspring.append(child2)

        combined_population = population + offspring
        combined_fitness = [multiobjective_func(ind) for ind in combined_population]

        fronts = non_dominated_sort(combined_fitness)
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) > population_size:
                distances = crowding_distance(front, combined_fitness)
                sorted_indices = np.argsort(-distances)
                front = [front[i] for i in sorted_indices]
                new_population.extend(front[:population_size - len(new_population)])
                break
            new_population.extend(front)

        population = [combined_population[i] for i in new_population]
        fitness = [combined_fitness[i] for i in new_population]

    return population, fitness

# Prepare all 20 functions
functions = [
    ("1. Ackley", ackley, [(-5, 5)] * 2),
    ("2. Booth", booth, [(-5, 5)] * 2),
    ("3. Rastrigin", rastrigin, [(-5, 5)] * 2),
    ("4. Rosenbrock", rosenbrock, [(-5, 5)] * 2),
    ("5. Schwefel", schwefel, [(-500, 500)] * 2),
    ("6. Sphere", sphere, [(-5, 5)] * 2),
    ("7. Michalewicz", michalewicz, [(0, np.pi)] * 2),
    ("8. Zakharov", zakharov, [(-5, 5)] * 2),
    ("9. Eggholder", eggholder, [(-512, 512)] * 2),
    ("10. Beale", beale, [(-4.5, 4.5)] * 2),
    ("11. Trid", trid, [(-5, 5)] * 2),
    ("12. Dixon-Price", dixon_price, [(-5, 5)] * 2),
    ("13. Cross-in-Tray", cross_in_tray, [(-10, 10)] * 2),
    ("14. Griewank", griewank, [(-600, 600)] * 2),
    ("15. Levy", levy, [(-10, 10)] * 2),
    ("16. Matyas", matyas, [(-10, 10)] * 2),
    ("17. Goldstein-Price", goldstein_price, [(-2, 2)] * 2),
    ("18. Powell", powell, [(-5, 5)] * 4),
    ("19. Bird", bird, [(-2 * np.pi, 2 * np.pi)] * 2),
    ("20. Pyramid", pyramid, [(-5, 5)] * 2)
]

# Prepare the plot
fig, axes = plt.subplots(4, 5, figsize=(20, 20))
axes = axes.ravel()

# Run NSGA-II and display results for all functions
for idx, (name, func, bounds) in enumerate(functions):
    print(f"\nRunning {name}...")

    start_time = time.time()
    memory_before = memory_usage()[0]

    population, fitness = nsga2(func, bounds, population_size=50, generations=100)

    memory_after = memory_usage()[0]
    end_time = time.time()

    print(f"Function: {name}")
    print(f"Convergence Time: {end_time - start_time:.2f} seconds")
    print(f"Memory Usage: {max(0, memory_after - memory_before):.2f} MB")
    print(f"Complexity Class: O(n^2 * g)")

    pareto_front = np.array([f for f in fitness])
    axes[idx].scatter(pareto_front[:, 0], pareto_front[:, 1], c='blue')
    axes[idx].set_title(name)
    axes[idx].set_xlabel("Objective 1")
    axes[idx].set_ylabel("Objective 2")

plt.tight_layout()
plt.show()
