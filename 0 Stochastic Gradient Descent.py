import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import time

# Define benchmark functions
def ackley(x):
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.exp(1)

def booth(x):
    return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def schwefel(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def sphere(x):
    return np.sum(x**2)

def michalewicz(x):
    m = 10
    return -np.sum(np.sin(x) * np.sin(((np.arange(len(x)) + 1) * x**2) / np.pi)**(2 * m))

def zakharov(x):
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * (np.arange(len(x)) + 1) * x)
    return sum1 + sum2**2 + sum2**4

def eggholder(x):
    return -(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1] + 47)))) - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47))))

def beale(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

def trid(x):
    return np.sum((x - 1)**2) - np.sum(x[:-1] * x[1:])

def dixon_price(x):
    return (x[0] - 1)**2 + np.sum([(i + 1) * (2 * x[i]**2 - x[i-1])**2 for i in range(1, len(x))])

def cross_in_tray(x):
    fact1 = np.sin(x[0]) * np.sin(x[1])
    fact2 = np.exp(abs(100 - np.sqrt(x[0]**2 + x[1]**2) / np.pi))
    return -0.0001 * (abs(fact1 * fact2) + 1)**0.1

def griewank(x):
    return 1 + np.sum(x**2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

def levy(x):
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def matyas(x):
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

def goldstein_price(x):
    term1 = 1 + ((x[0] + x[1] + 1)**2) * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
    term2 = 30 + ((2*x[0] - 3*x[1])**2) * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2)
    return term1 * term2

def powell(x):
    term1 = (x[0] + 10*x[1])**2
    term2 = 5 * (x[2] - x[3])**2
    term3 = (x[1] - 2*x[2])**4
    term4 = 10 * (x[0] - x[3])**4
    return term1 + term2 + term3 + term4

def bird(x):
    return np.sin(x[0]) * np.exp((1 - np.cos(x[1]))**2) + np.cos(x[1]) * np.exp((1 - np.sin(x[0]))**2) + (x[0] - x[1])**2

def pyramid(x):
    return np.sum(np.abs(x))

def numerical_gradient(func, x, epsilon=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x1, x2 = x.copy(), x.copy()
        x1[i] += epsilon
        x2[i] -= epsilon
        grad[i] = (func(x1) - func(x2)) / (2 * epsilon)
    return grad

def stochastic_gradient_descent(func, x0, lr=0.01, max_iter=200):
    x = np.array(x0)
    costs = []
    for _ in range(max_iter):
        grad = numerical_gradient(func, x)
        # Choose a random dimension to update (stochastic part)
        idx = np.random.randint(0, len(x))
        x[idx] -= lr * grad[idx]
        costs.append(func(x))
    return x, costs

# Prepare 20 functions
functions = [
    ("1. Ackley", ackley, np.random.uniform(-5, 5, 2)),
    ("2. Booth", booth, np.random.uniform(-5, 5, 2)),
    ("3. Rastrigin", rastrigin, np.random.uniform(-5, 5, 2)),
    ("4. Rosenbrock", rosenbrock, np.random.uniform(-5, 5, 2)),
    ("5. Schwefel", schwefel, np.random.uniform(-500, 500, 2)),
    ("6. Sphere", sphere, np.random.uniform(-5, 5, 2)),
    ("7. Michalewicz", michalewicz, np.random.uniform(0, np.pi, 2)),
    ("8. Zakharov", zakharov, np.random.uniform(-5, 5, 2)),
    ("9. Eggholder", eggholder, np.random.uniform(-512, 512, 2)),
    ("10. Beale", beale, np.random.uniform(-4.5, 4.5, 2)),
    ("11. Trid", trid, np.random.uniform(-5, 5, 2)),
    ("12. Dixon-Price", dixon_price, np.random.uniform(-5, 5, 2)),
    ("13. Cross-in-Tray", cross_in_tray, np.random.uniform(-10, 10, 2)),
    ("14. Griewank", griewank, np.random.uniform(-600, 600, 2)),
    ("15. Levy", levy, np.random.uniform(-10, 10, 2)),
    ("16. Matyas", matyas, np.random.uniform(-10, 10, 2)),
    ("17. Goldstein-Price", goldstein_price, np.random.uniform(-2, 2, 2)),
    ("18. Powell", powell, np.random.uniform(-5, 5, 4)),
    ("19. Bird", bird, np.random.uniform(-2 * np.pi, 2 * np.pi, 2)),
    ("20. Pyramid", pyramid, np.random.uniform(-5, 5, 2))
]

# Prepare the plot
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
axes = axes.ravel()

# Run Stochastic Gradient Descent and display results for all functions
for idx, (name, func, x0) in enumerate(functions):
    print(f"\nRunning {name}...")

    start_time = time.time()
    memory_before = memory_usage()[0]

    best_x, costs = stochastic_gradient_descent(func, x0, lr=0.01, max_iter=100)

    memory_after = memory_usage()[0]
    end_time = time.time()

    print(f"Function: {name}")
    print(f"Best Cost: {costs[-1] if costs else 'N/A'}")
    print(f"Convergence Time: {end_time - start_time} seconds")
    print(f"Memory Usage: {max(0, memory_after - memory_before)} MB")
    print("Complexity Class: O(n * d)")

    if idx < len(axes):
        axes[idx].plot(costs)
        axes[idx].set_title(name)
        axes[idx].set_xlabel("Iterations")
        axes[idx].set_ylabel("Cost")

plt.tight_layout()
plt.show()
