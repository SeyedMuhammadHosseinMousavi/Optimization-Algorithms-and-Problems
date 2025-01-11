import numpy as np
import matplotlib.pyplot as plt

# Particle Swarm Optimization (PSO) Parameters
population_size = 20
num_iterations = 50
inertia = 0.5
c1 = 2.0  # Cognitive parameter
c2 = 2.0  # Social parameter
image_resolution = (300, 300)

# Fitness Function: Enhanced to reward structured patterns

def fitness_function(image):
    complexity = calculate_complexity(image)
    symmetry = calculate_symmetry(image)
    contrast = global_contrast_factor(image)
    color_diversity = calculate_color_diversity(image)
    return complexity + symmetry + contrast + color_diversity

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

# Calculate Color Diversity
def calculate_color_diversity(image):
    unique_colors = len(np.unique(image.reshape(-1, image.shape[2]), axis=0))
    return unique_colors / (image.shape[0] * image.shape[1])

# Initialize Population with fractal-like patterns and colorful variations
def initialize_population(size, resolution):
    population = []
    for _ in range(size):
        x = np.linspace(-2.0, 2.0, resolution[0])
        y = np.linspace(-2.0, 2.0, resolution[1])
        x, y = np.meshgrid(x, y)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        fractal_r = (np.sin(10 * r + 5 * theta) * 127 + 128).astype(np.uint8)
        fractal_g = (np.cos(10 * r - 5 * theta) * 127 + 128).astype(np.uint8)
        fractal_b = ((np.sin(10 * theta) + np.cos(10 * r)) * 127 + 128).astype(np.uint8)
        image = np.stack([fractal_r, fractal_g, fractal_b], axis=2)
        population.append(image)
    return population

# PSO Update Function
def update_particles(positions, velocities, personal_best_positions, global_best_position, inertia, c1, c2):
    for i in range(len(positions)):
        r1, r2 = np.random.random(), np.random.random()
        cognitive_component = c1 * r1 * (personal_best_positions[i] - positions[i])
        social_component = c2 * r2 * (global_best_position - positions[i])
        velocities[i] = inertia * velocities[i] + cognitive_component + social_component
        positions[i] = np.clip(positions[i] + velocities[i], 0, 255)  # Keep positions within valid range

# Main PSO Loop
population = initialize_population(population_size, image_resolution)
velocities = [np.random.uniform(-1, 1, (image_resolution[0], image_resolution[1], 3)) for _ in range(population_size)]
personal_best_positions = population[:]
personal_best_scores = [fitness_function(image) for image in population]
global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
global_best_score = max(personal_best_scores)

for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}: Evaluating fitness")

    # Evaluate Fitness
    fitness_scores = [fitness_function(image) for image in population]

    for i in range(population_size):
        if fitness_scores[i] > personal_best_scores[i]:
            personal_best_scores[i] = fitness_scores[i]
            personal_best_positions[i] = population[i]

    best_particle_index = np.argmax(personal_best_scores)
    if personal_best_scores[best_particle_index] > global_best_score:
        global_best_score = personal_best_scores[best_particle_index]
        global_best_position = personal_best_positions[best_particle_index]

    print(f"  Best fitness: {global_best_score:.2f}")

    # Update Particles
    update_particles(population, velocities, personal_best_positions, global_best_position, inertia, c1, c2)

    # Display the best image
    plt.imshow(global_best_position.astype(np.uint8))
    plt.title(f"Iteration {iteration + 1} - Best Fitness: {global_best_score:.2f}")
    plt.axis('off')
    plt.pause(0.5)

print("Intelligent art generation with PSO complete!")
