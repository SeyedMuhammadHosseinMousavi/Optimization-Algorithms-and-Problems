import numpy as np
import matplotlib.pyplot as plt

# Objective function
def objective_function(x, start, end, lambda_bend, warp_field):
    # Distance term (Geodesic distance using warp field)
    distance = np.linalg.norm((x - end) * (1 + warp_field))  # Adjusted distance with warp effect

    # Bending cost (simulating space-time distortion effort)
    bending_cost = lambda_bend * np.sum((x - start)**2)  # Quadratic bending penalty

    # Effort to traverse (simulate time dilation or warp effort)
    effort_cost = np.sum(np.abs(x - start) * (1 + warp_field))  # Absolute effort adjusted by warp

    # Energy cost for maintaining the warp field
    energy_cost = np.sum(warp_field**2)  # Energy required to maintain the warp

    # Combined cost function
    return distance + bending_cost + 0.5 * effort_cost + 0.2 * energy_cost

# Parameters for Firefly Algorithm
num_fireflies = 70
num_dimensions = 2
num_iterations = 40
start = np.array([0, 0])
end = np.array([10, 10])
lambda_bend = 0.1
warp_field = np.random.uniform(low=0.1, high=0.5, size=num_dimensions)  # Random initial warp field
alpha = 0.2  # Randomness strength
beta0 = 1.0  # Base attractiveness
gamma = 1.0  # Absorption coefficient

# Initialize fireflies
positions = np.random.uniform(low=-5, high=15, size=(num_fireflies, num_dimensions))
intensities = np.array([objective_function(p, start, end, lambda_bend, warp_field) for p in positions])

# Record the best cost at each iteration for plotting
best_costs = []

# Optimization loop
for iteration in range(num_iterations):
    for i in range(num_fireflies):
        for j in range(num_fireflies):
            if intensities[j] < intensities[i]:  # Move firefly i towards firefly j
                distance = np.linalg.norm(positions[i] - positions[j])
                beta = beta0 * np.exp(-gamma * distance**2)
                positions[i] += beta * (positions[j] - positions[i]) + alpha * (np.random.rand(num_dimensions) - 0.5)

        # Update intensity for firefly i
        intensities[i] = objective_function(positions[i], start, end, lambda_bend, warp_field)

    # Find the best firefly
    best_idx = np.argmin(intensities)
    best_costs.append(intensities[best_idx])

    # Print progress
    print(f"Iteration {iteration + 1}: Best Fitness = {intensities[best_idx]}")

# Final results
best_position = positions[best_idx]
print("\nOptimization Completed!")
print(f"Global Best Position: {best_position}")
print(f"Objective Value at Global Best: {intensities[best_idx]}")

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(best_costs, label="Best Cost per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Convergence of Firefly Algorithm with Space-Time Bending Analogy")
plt.legend()
plt.grid()
plt.show()

# Plot the final positions of fireflies
plt.figure(figsize=(8, 8))
plt.scatter(positions[:, 0], positions[:, 1], label="Final Firefly Positions", color="blue")
plt.scatter(best_position[0], best_position[1], label="Global Best Position", color="red", marker="x", s=100)
plt.scatter(end[0], end[1], label="Target Position", color="green", marker="*", s=200)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Final Firefly Distribution")
plt.legend()
plt.grid()
plt.show()
