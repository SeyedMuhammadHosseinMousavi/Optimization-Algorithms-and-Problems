import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to generate a random but feasible resource allocation problem
def generate_feasible_resource_allocation_problem(num_resources, num_tasks):
    task_demands = np.random.randint(20, 60, size=num_tasks)  # Larger demands
    total_demand = np.sum(task_demands)
    resource_capacities = np.random.randint(50, 100, size=num_resources)  # Generate capacities
    resource_capacities *= (total_demand // np.sum(resource_capacities) + 1)  # Scale capacities to ensure feasibility
    cost_matrix = np.random.randint(10, 50, size=(num_tasks, num_resources))  # Higher cost range
    return resource_capacities, task_demands, cost_matrix

# Objective function: calculate total cost of allocation
def calculate_total_cost(allocation, cost_matrix):
    total_cost = 0
    for task, resource in enumerate(allocation):
        total_cost += cost_matrix[task, resource]
    return total_cost

# Check feasibility of an allocation
def is_feasible(allocation, task_demands, resource_capacities):
    allocated_resources = np.zeros(len(resource_capacities))
    for task, resource in enumerate(allocation):
        allocated_resources[resource] += task_demands[task]
    return np.all(allocated_resources <= resource_capacities)

# Stochastic Gradient Descent (SGD) for Resource Allocation
def stochastic_gradient_descent(task_demands, resource_capacities, cost_matrix, learning_rate=0.1, max_iterations=200):
    num_tasks, num_resources = cost_matrix.shape
    
    # Initialize allocation as random (one resource per task)
    allocation = np.random.randint(0, num_resources, size=num_tasks)
    
    def calculate_gradient(allocation):
        # Calculate gradient of the cost function (approximation)
        gradient = np.zeros_like(allocation, dtype=float)
        for task in range(num_tasks):
            current_resource = allocation[task]
            costs = cost_matrix[task, :]
            gradient[task] = costs[current_resource] - np.min(costs)
        return gradient
    
    def project_to_feasible(allocation):
        # Project allocation to feasible space (ensure capacity constraints)
        allocated_resources = np.zeros(num_resources)
        for task, resource in enumerate(allocation):
            allocated_resources[resource] += task_demands[task]
        
        for task, resource in enumerate(allocation):
            if allocated_resources[resource] > resource_capacities[resource]:
                # Reallocate task to a feasible resource
                feasible_resources = [r for r in range(num_resources) if allocated_resources[r] + task_demands[task] <= resource_capacities[r]]
                if feasible_resources:
                    new_resource = np.random.choice(feasible_resources)
                    allocated_resources[resource] -= task_demands[task]
                    allocated_resources[new_resource] += task_demands[task]
                    allocation[task] = new_resource
        return allocation
    
    cost_progress = []
    for iteration in range(max_iterations):
        # Calculate cost and gradient
        current_cost = calculate_total_cost(allocation, cost_matrix)
        gradient = calculate_gradient(allocation)
        
        # Update allocation using SGD step
        allocation = allocation - learning_rate * gradient
        allocation = np.round(np.clip(allocation, 0, num_resources - 1)).astype(int)
        
        # Project back to feasible space
        allocation = project_to_feasible(allocation)
        
        # Track progress
        current_cost = calculate_total_cost(allocation, cost_matrix)
        cost_progress.append(current_cost)
    
    # Return the best solution and cost progress
    return allocation, current_cost, cost_progress

# Main pipeline execution
def main_pipeline(num_resources=6, num_tasks=10, learning_rate=0.03, max_iterations=100):
    # Generate a new random problem
    resource_capacities, task_demands, cost_matrix = generate_feasible_resource_allocation_problem(num_resources, num_tasks)
    
    # Solve using SGD
    best_allocation, best_cost, cost_progress = stochastic_gradient_descent(
        task_demands, resource_capacities, cost_matrix, learning_rate, max_iterations
    )
    
    # Plotting fitness progress
    plt.figure(figsize=(10, 6))
    plt.plot(cost_progress, marker="o", linestyle="--")
    plt.title("Fitness over Iterations - Stochastic Gradient Descent")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (Total Cost)")
    plt.grid()
    plt.show()

    # Displaying problem and results
    results = pd.DataFrame({
        "Task": np.arange(1, num_tasks + 1),
        "Allocated Resource": best_allocation,
        "Task Demand": task_demands,
        "Resource Capacity": [resource_capacities[res] for res in best_allocation],
        "Cost": [cost_matrix[task, best_allocation[task]] for task in range(num_tasks)],
    })
    summary = pd.DataFrame({
        "Total Cost": [best_cost],
        "Feasible": [is_feasible(best_allocation, task_demands, resource_capacities)]
    })

    # Save and display results
    results.to_csv("Resource_Allocation_Results.csv", index=False)
    summary.to_csv("Resource_Allocation_Summary.csv", index=False)

    print("Detailed Results:")
    print(results)
    print("\nSummary:")
    print(summary)

# Run the pipeline
main_pipeline()
