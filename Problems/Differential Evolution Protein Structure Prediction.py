import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import splprep, splev

# Define the Enhanced Energy Function
def energy_function(positions):
    """Calculate the energy with Lennard-Jones potential and harmonic bonds."""
    distances = pdist(positions)  # Pairwise distances
    distances_matrix = squareform(distances)

    # Lennard-Jones potential for non-adjacent residues
    lj_energy = np.sum(4 * ((1 / distances[distances > 0])**12 - (1 / distances[distances > 0])**6))

    # Harmonic bond potential for adjacent residues
    bond_energy = 0.5 * np.sum((distances_matrix[np.arange(len(positions)-1), np.arange(1, len(positions))] - 1)**2)

    return lj_energy + bond_energy

# Initialize DE Parameters
num_particles = 40
num_dimensions = 3  # 3D space
num_amino_acids = 15  # Number of residues in the protein
num_iterations = 400

# Differential Evolution hyperparameters
mutation_factor = 0.5  # Controls the step size
crossover_probability = 0.9  # Probability of crossover

# Initialize particle positions
positions = np.random.uniform(-5, 5, (num_particles, num_amino_acids, num_dimensions))

# Evaluate initial fitness
fitness_scores = np.array([energy_function(p) for p in positions])

# Track convergence
convergence = []

# DE Main Loop
for iteration in range(num_iterations):
    for i in range(num_particles):
        # Mutation: Create a donor vector
        indices = np.random.choice(np.delete(np.arange(num_particles), i), 3, replace=False)
        a, b, c = positions[indices]
        donor_vector = a + mutation_factor * (b - c)

        # Crossover: Create a trial vector
        trial_vector = np.copy(positions[i])
        for j in range(num_amino_acids):
            if np.random.rand() < crossover_probability:
                trial_vector[j] = donor_vector[j]

        # Selection: Compare trial vector with target vector
        trial_fitness = energy_function(trial_vector)
        if trial_fitness < fitness_scores[i]:
            positions[i] = trial_vector
            fitness_scores[i] = trial_fitness

    # Track global best
    global_best_index = np.argmin(fitness_scores)
    global_best_position = positions[global_best_index]
    global_best_score = fitness_scores[global_best_index]

    # Track convergence
    convergence.append(global_best_score)
    print(f"Iteration {iteration + 1}/{num_iterations}, Best Score: {global_best_score:.4f}")

# Plot the convergence
plt.figure(figsize=(12, 6))
plt.plot(convergence, marker='o', linewidth=2)
plt.title("Convergence of DE on Enhanced Energy Function")
plt.xlabel("Iteration")
plt.ylabel("Best Energy")
plt.grid()
plt.show()

# Visualize the final protein structure
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Smooth the backbone with splines
tck, u = splprep([global_best_position[:, 0], global_best_position[:, 1], global_best_position[:, 2]], s=2)
smoothed_coords = splev(np.linspace(0, 1, 100), tck)

# Plot amino acids
ax.scatter(global_best_position[:, 0], global_best_position[:, 1], global_best_position[:, 2], c='r', s=100, label='Amino Acids')

# Plot smoothed backbone
ax.plot(smoothed_coords[0], smoothed_coords[1], smoothed_coords[2], c='b', linewidth=2, label='Backbone')

# Annotate amino acids
for i, (x, y, z) in enumerate(global_best_position):
    ax.text(x, y, z, str(i), color='black', fontsize=10)

ax.set_title("Optimized Protein Structure with Enhanced DE")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()

# Print final details
print("Final Optimized Amino Acid Positions:")
print(global_best_position)
print(f"Final Optimized Energy: {global_best_score:.4f}")
