import numpy as np
import matplotlib.pyplot as plt

# Genetic Algorithm Parameters
population_size = 30
num_generations = 100
mutation_rate = 0.1
image_resolution = (300, 300)

# Fitness Function: Enhanced to reward spiral-like patterns and symmetry
def fitness_function(image):
    complexity = calculate_complexity(image)
    symmetry = calculate_symmetry(image)
    contrast = global_contrast_factor(image)
    return complexity + symmetry + contrast

# Calculate Complexity using edge detection
def calculate_complexity(image):
    gradient_x = np.abs(np.diff(image, axis=0))
    gradient_y = np.abs(np.diff(image, axis=1))
    complexity = np.sum(gradient_x) + np.sum(gradient_y)
    return complexity / image.size

# Global Contrast Factor (GCF)
def global_contrast_factor(image):
    luminance = np.mean(image, axis=2)  # Average across RGB channels
    contrast = np.std(luminance)  # Standard deviation as a simple contrast measure
    return contrast

# Calculate Symmetry
def calculate_symmetry(image):
    vertical_symmetry = np.sum(np.abs(image - np.flip(image, axis=1)))
    horizontal_symmetry = np.sum(np.abs(image - np.flip(image, axis=0)))
    total_symmetry = -(vertical_symmetry + horizontal_symmetry) / image.size  # Negate to reward symmetry
    return total_symmetry

# Initialize Population with spiral-like patterns
def initialize_population(size, resolution):
    population = []
    for _ in range(size):
        x = np.linspace(-1.0, 1.0, resolution[0])
        y = np.linspace(-1.0, 1.0, resolution[1])
        x, y = np.meshgrid(x, y)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        spiral = (np.sin(10 * r + 5 * theta) * 127 + 128).astype(np.uint8)
        image = np.stack([spiral, spiral, spiral], axis=2)  # Grayscale to RGB
        population.append(image)
    return population

# Crossover Operation
def crossover(parent1, parent2):
    crossover_point = np.random.randint(0, parent1.shape[1])
    child = np.concatenate((parent1[:, :crossover_point], parent2[:, crossover_point:]), axis=1)
    return child

# Mutation Operation (structured fractal adjustments)
def mutate(image, rate):
    mutated_image = image.copy()
    num_pixels = np.prod(image.shape[:2])
    num_mutations = int(rate * num_pixels)
    for _ in range(num_mutations):
        x, y = np.random.randint(0, image.shape[0]), np.random.randint(0, image.shape[1])
        mutated_image[x, y] = np.clip(mutated_image[x, y] + np.random.randint(-50, 50), 0, 255)
    return mutated_image

# Main GA Loop
population = initialize_population(population_size, image_resolution)
for generation in range(num_generations):
    print(f"Generation {generation + 1}: Evaluating fitness")

    # Evaluate Fitness
    fitness_scores = [fitness_function(image) for image in population]
    best_fitness = max(fitness_scores)
    print(f"  Best fitness: {best_fitness:.2f}")

    # Select Parents (Roulette Wheel Selection)
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    indices = np.arange(len(population))  # Indices for the population
    selected_indices = np.random.choice(indices, size=population_size, p=probabilities, replace=True)
    parents = [population[i] for i in selected_indices]

    print("  Parents selected")

    # Generate New Population
    new_population = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[(i + 1) % len(parents)]
        child = crossover(parent1, parent2)
        child = mutate(child, mutation_rate)
        new_population.append(child)
    population = new_population

    print("  New population generated")

    # Safely find the best image
    best_index = np.argmax(fitness_scores)
    best_image = population[best_index]

    # Display the best image
    plt.imshow(best_image, cmap='inferno')
    plt.title(f"Generation {generation + 1} - Best Fitness: {best_fitness:.2f}")
    plt.axis('off')
    plt.pause(0.5)

print("Evolutionary art generation complete!")