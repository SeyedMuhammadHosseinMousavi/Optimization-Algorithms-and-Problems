import numpy as np
import matplotlib.pyplot as plt

# Define the objective function
def objective_function(antenna_points):
    """
    Objective Function for Antenna Design Optimization

    This function evaluates the quality of an antenna design by combining multiple
    factors such as:
    - Total length of the antenna (to ensure it remains compact and efficient).
    - Smoothness, penalizing excessive twists and encouraging smooth transitions.

    Parameters:
    antenna_points (numpy.ndarray): Array of 3D coordinates representing the antenna geometry.

    Returns:
    float: The computed cost for the given antenna design.
    """
    # Compute total length of the antenna
    total_length = np.sum(np.sqrt(np.sum(np.diff(antenna_points, axis=0)**2, axis=1)))

    # Compute smoothness penalty (penalize excessive variation in angles between segments)
    smoothness_penalty = np.sum(np.abs(np.diff(antenna_points[:, 2])))

    # Combine metrics into the cost function
    cost = total_length + 0.3 * smoothness_penalty
    return cost

# Function to generate initial antenna with seven joints
def generate_initial_antenna():
    """
    Generate an initial antenna geometry with seven joints.

    Returns:
    numpy.ndarray: Array of 3D coordinates representing the initial antenna geometry.
    """
    joints = 7
    z = np.linspace(0, 10, joints + 1)  # Antenna progresses upward
    x = np.random.uniform(-1, 1, joints + 1)
    y = np.random.uniform(-1, 1, joints + 1)
    return np.column_stack((x, y, z))

# PSO Parameters
num_particles = 30
num_iterations = 200
joints = 7  # Number of joints
dimensions = joints * 3  # 3D coordinates for each joint

# Function to run PSO and return results
def run_pso():
    # Initialize particle positions and velocities
    particles = np.random.uniform(-1, 1, (num_particles, dimensions))
    velocities = np.random.uniform(-0.1, 0.1, (num_particles, dimensions))
    best_particle_positions = particles.copy()
    best_particle_costs = np.array([objective_function(p.reshape(-1, 3)) for p in particles])
    global_best_position = particles[np.argmin(best_particle_costs)]
    global_best_cost = np.min(best_particle_costs)

    # PSO Hyperparameters
    w = 0.5  # Inertia weight
    c1 = 1.5  # Cognitive coefficient
    c2 = 1.5  # Social coefficient

    # Track cost over iterations
    cost_history = []

    # PSO Main Loop
    for iteration in range(num_iterations):
        for i, particle in enumerate(particles):
            # Update velocity
            r1, r2 = np.random.random(2)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (best_particle_positions[i] - particle) +
                             c2 * r2 * (global_best_position - particle))

            # Update position
            particles[i] += velocities[i]

            # Constrain particles within bounds
            particles[i] = np.clip(particles[i], -1, 1)

            # Evaluate cost
            reshaped_particle = particles[i].reshape(-1, 3)
            cost = objective_function(reshaped_particle)

            # Update personal best
            if cost < best_particle_costs[i]:
                best_particle_costs[i] = cost
                best_particle_positions[i] = particles[i]

            # Update global best
            if cost < global_best_cost:
                global_best_cost = cost
                global_best_position = particles[i]

        cost_history.append(global_best_cost)

    return global_best_position.reshape(-1, 3), cost_history

# Plot 4 antennas and their costs in a 2x4 layout
fig, axes = plt.subplots(2, 4, figsize=(20, 10), subplot_kw={})

for i in range(4):
    best_antenna_points, cost_history = run_pso()

    # Plot antenna geometry (3D plot)
    ax = fig.add_subplot(2, 4, i + 1, projection='3d')
    ax.plot(best_antenna_points[:, 0], best_antenna_points[:, 1], best_antenna_points[:, 2], marker='o', linewidth=2)
    ax.set_title(f"Optimized Antenna {i + 1}", fontsize=14)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)
    ax.scatter(best_antenna_points[0, 0], best_antenna_points[0, 1], best_antenna_points[0, 2], color='red', label='Start', s=100)
    ax.scatter(best_antenna_points[-1, 0], best_antenna_points[-1, 1], best_antenna_points[-1, 2], color='green', label='End', s=100)
    ax.legend(fontsize=10)

    # Plot cost history (2D plot)
    ax2 = fig.add_subplot(2, 4, i + 5)
    ax2.plot(range(1, num_iterations + 1), cost_history, marker='o', color='blue', linewidth=2)
    ax2.set_title(f"Cost Over Iterations {i + 1}", fontsize=14)
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Cost", fontsize=12)

plt.tight_layout()
plt.show()
