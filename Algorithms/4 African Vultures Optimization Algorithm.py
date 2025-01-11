import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import time

# Define benchmark functions
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
    return np.sum(np.abs(x) * np.sin(x))

# TLBO Algorithm
def tlbo(func, bounds, population_size=50, generations=100):
    dimensions = len(bounds)

# African Vultures Optimization Algorithm (AVOA)
def avoa(func, bounds, population_size=50, generations=100):
    dimensions = len(bounds)

    # Initialize the population
    vultures = [np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], dimensions) for _ in range(population_size)]
    fitness = [func(vulture) for vulture in vultures]

    # Find the best and worst solutions
    best_vulture = vultures[np.argmin(fitness)]
    best_cost = min(fitness)
    worst_vulture = vultures[np.argmax(fitness)]
    worst_cost = max(fitness)
    cost_history = []

    for generation in range(generations):
        a = 2 - generation * (2 / generations)  # Linearly decreasing parameter
        for i in range(population_size):
            r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
            A = 2 * a * r1 - a  # Exploration parameter
            C = 2 * r2          # Influence parameter

            if np.random.rand() < 0.5:
                # Exploration phase
                D = abs(C * best_vulture - vultures[i])
                new_position = best_vulture - A * D
            else:
                # Exploitation phase
                D1 = abs(C * best_vulture - vultures[i])
                D2 = abs(C * worst_vulture - vultures[i])
                new_position = (best_vulture + worst_vulture) / 2 - A * ((D1 + D2) / 2)

            # Clip the new position within bounds
            new_position = np.clip(new_position, [b[0] for b in bounds], [b[1] for b in bounds])
            new_fitness = func(new_position)

            # Update vulture position if fitness improves
            if new_fitness < fitness[i]:
                vultures[i] = new_position
                fitness[i] = new_fitness

        # Update the best and worst solutions
        generation_best_cost = min(fitness)
        if generation_best_cost < best_cost:
            best_vulture = vultures[np.argmin(fitness)]
            best_cost = generation_best_cost

        worst_cost = max(fitness)
        worst_vulture = vultures[np.argmax(fitness)]

        cost_history.append(best_cost)

    return best_vulture, cost_history

# Benchmark functions and bounds
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
# Complexity Mapping Function (if required)
def map_complexity(n, g):
    complexity_classes = {
        "O(1)": 1,
        "O(log n)": lambda n: np.log2(n),
        "O(n)": lambda n: n,
        "O(n log n)": lambda n: n * np.log2(n),
        "O(n^2)": lambda n: n**2,
        "O(n^3)": lambda n: n**3,
        "O(2^n)": lambda n: 2**n,
        "O(n!)": lambda n: np.math.factorial(n)
    }
    complexity = n * g
    for label, func in complexity_classes.items():
        if callable(func) and complexity <= func(n):
            return label
    return "O(n^2)"

# Run AVOA and visualize results
fig, axes = plt.subplots(4, 5, figsize=(20, 20))
axes = axes.ravel()

for idx, (name, func, bounds) in enumerate(functions):
    print(f"\nRunning {name}...")

    start_time = time.time()
    memory_before = memory_usage()[0]

    best_x, costs = avoa(func, bounds, population_size=50, generations=100)

    memory_after = memory_usage()[0]
    end_time = time.time()

    complexity_class = map_complexity(50, 100)

    print(f"Function: {name}")
    print(f"Best Cost: {costs[-1]:.10f}")
    print(f"Convergence Time: {end_time - start_time:.10f} seconds")
    print(f"Memory Usage: {memory_after - memory_before:.10f} MB")
    print(f"Complexity Class (standard): {complexity_class}")
    print(f"Complexity (previous form): O(n * g)")

    if idx < len(axes):
        axes[idx].plot(costs)
        axes[idx].set_title(name)
        axes[idx].set_xlabel("Generations")
        axes[idx].set_ylabel("Cost")

plt.tight_layout()
plt.show()
