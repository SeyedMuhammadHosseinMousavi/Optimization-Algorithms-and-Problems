import numpy as np
import matplotlib.pyplot as plt
import random

# Function to generate a random parallel machine scheduling problem
def generate_parallel_machine_problem(num_tasks, num_machines):
    tasks = np.arange(1, num_tasks + 1)  # Task IDs
    processing_times = np.random.randint(10, 100, size=num_tasks)  # Random processing times
    return tasks, processing_times, num_machines

# Objective function: calculate makespan (Cmax) of a given schedule
def calculate_makespan(schedule, processing_times, num_machines):
    machine_times = np.zeros(num_machines)
    for machine, task in enumerate(schedule):
        machine_times[machine % num_machines] += processing_times[task - 1]
    return max(machine_times)

# Brain Storm Optimization (BSO) for Parallel Machine Scheduling
def brain_storm_optimization(tasks, processing_times, num_machines, iterations=200, population_size=100):
    num_tasks = len(tasks)
    best_schedule = None
    best_makespan = float('inf')
    
    # Initial population of random schedules
    population = [np.random.permutation(tasks) for _ in range(population_size)]
    makespans = [calculate_makespan(schedule, processing_times, num_machines) for schedule in population]
    
    best_schedule = population[np.argmin(makespans)]
    best_makespan = min(makespans)
    
    makespan_progress = [best_makespan]
    
    for iteration in range(iterations):
        # Generate new solutions by mutation and combination
        new_population = []
        for i in range(population_size):
            if random.random() < 0.5:  # Mutation
                new_schedule = population[i].copy()
                idx1, idx2 = np.random.choice(num_tasks, 2, replace=False)
                new_schedule[idx1], new_schedule[idx2] = new_schedule[idx2], new_schedule[idx1]
            else:  # Combination
                parent1, parent2 = random.sample(population, 2)
                split_point = np.random.randint(1, num_tasks - 1)
                new_schedule = np.concatenate((parent1[:split_point], parent2[split_point:]))
                new_schedule, _ = np.unique(new_schedule, return_index=True)
                new_schedule = np.append(new_schedule, np.setdiff1d(tasks, new_schedule))
            
            new_population.append(new_schedule)
        
        # Evaluate new population
        new_makespans = [calculate_makespan(schedule, processing_times, num_machines) for schedule in new_population]
        
        # Update best solution
        min_new_makespan = min(new_makespans)
        if min_new_makespan < best_makespan:
            best_makespan = min_new_makespan
            best_schedule = new_population[np.argmin(new_makespans)]
        
        # Replace old population with new one
        population = new_population
        makespan_progress.append(best_makespan)
    
    return best_schedule, best_makespan, makespan_progress

# Main execution for a single run
num_tasks = 14
num_machines = 4
tasks, processing_times, num_machines = generate_parallel_machine_problem(num_tasks, num_machines)

best_schedule, best_makespan, makespan_progress = brain_storm_optimization(
    tasks, processing_times, num_machines
)

# Plotting iteration progress
plt.figure(figsize=(10, 6))
plt.plot(makespan_progress, marker="o", linestyle="--")
plt.title("Makespan over Iterations - Brain Storm Optimization")
plt.xlabel("Iteration")
plt.ylabel("Makespan (Cmax)")
plt.grid()
plt.show()

# Plotting the solution similar to the provided image
machine_assignments = [[] for _ in range(num_machines)]
machine_times = np.zeros(num_machines)
for task in best_schedule:
    machine = np.argmin(machine_times)
    machine_assignments[machine].append(task)
    machine_times[machine] += processing_times[task - 1]

plt.figure(figsize=(12, 8))
for i, machine in enumerate(machine_assignments, 1):
    start = 0
    for task in machine:
        plt.barh(i, processing_times[task - 1], left=start, color="lime", edgecolor="black")
        plt.text(start + processing_times[task - 1] / 2, i, str(task), va='center', ha='center', fontsize=10, color="black")
        start += processing_times[task - 1]
plt.axvline(best_makespan, color="yellow", linestyle="--", linewidth=2, label=f"Cmax = {best_makespan}")
plt.title("Parallel Machine Scheduling")
plt.xlabel("Tasks")
plt.ylabel("Machines")
plt.yticks(range(1, num_machines + 1))
plt.legend()
plt.grid(axis="x")
plt.show()

# Display results
import pandas as pd
results = pd.DataFrame({
    "Best Schedule": [best_schedule],
    "Best Makespan": [best_makespan],
    "Iterations": [len(makespan_progress)],
})

# Display results in the console
print("Parallel Machine Scheduling Results:")
print(results)