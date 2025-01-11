import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

# Function to generate a random QAP problem
def generate_qap_problem(size):
    flow_matrix = np.random.randint(1, 100, size=(size, size))
    distance_matrix = np.random.randint(1, 100, size=(size, size))
    return flow_matrix, distance_matrix

# Objective function for QAP
def calculate_cost(permutation, flow_matrix, distance_matrix):
    size = len(permutation)
    cost = 0
    for i in range(size):
        for j in range(size):
            cost += flow_matrix[i][j] * distance_matrix[permutation[i]][permutation[j]]
    return cost

# Simulated Annealing for QAP
def simulated_annealing_qap(flow_matrix, distance_matrix, initial_temp=1000, cooling_rate=0.95, max_iterations=1000):
    size = len(flow_matrix)
    current_permutation = list(range(size))
    random.shuffle(current_permutation)
    current_cost = calculate_cost(current_permutation, flow_matrix, distance_matrix)
    
    best_permutation = current_permutation.copy()
    best_cost = current_cost
    
    costs_over_iterations = [current_cost]
    
    temperature = initial_temp
    
    for iteration in range(max_iterations):
        i, j = random.sample(range(size), 2)
        new_permutation = current_permutation.copy()
        new_permutation[i], new_permutation[j] = new_permutation[j], new_permutation[i]
        
        new_cost = calculate_cost(new_permutation, flow_matrix, distance_matrix)
        
        delta = new_cost - current_cost
        if delta < 0 or random.uniform(0, 1) < np.exp(-delta / temperature):
            current_permutation = new_permutation
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_permutation = current_permutation
                best_cost = current_cost
        
        costs_over_iterations.append(current_cost)
        temperature *= cooling_rate
    
    return best_permutation, best_cost, costs_over_iterations

# Main execution for a single run
size = 10  # Problem size
flow_matrix, distance_matrix = generate_qap_problem(size)
best_perm, best_cost, costs = simulated_annealing_qap(flow_matrix, distance_matrix)

# Plotting iteration cost progress
plt.figure(figsize=(10, 6))
plt.plot(costs, marker="o", linestyle="--")
plt.title("Cost over Iterations - Simulated Annealing")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid()
plt.show()

# Displaying the QAP flow matrix
plt.figure(figsize=(8, 6))
plt.imshow(flow_matrix, cmap="Blues", interpolation="nearest")
plt.colorbar()
plt.title("Flow Matrix")
plt.show()

# Displaying results
results = pd.DataFrame({
    "Best Permutation": [best_perm],
    "Best Cost": [best_cost],
    "Iterations": [len(costs)],
})
print("Simulated Annealing QAP Results:")
print(results)

# Save results to a CSV file
results.to_csv("Simulated_Annealing_QAP_Results.csv", index=False)
print("Results saved to 'Simulated_Annealing_QAP_Results.csv'")
