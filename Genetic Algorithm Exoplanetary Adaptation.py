import numpy as np
import matplotlib.pyplot as plt

# Generate random planet and its stellar system parameters
def generate_planet():
    """Generates random parameters for a hypothetical planet in a stellar system."""
    planet_name = f"Planet_{np.random.randint(1000, 9999)}"
    star_name = f"Star_{np.random.randint(1000, 9999)}"
    
    # Randomize planet parameters
    gravity = np.random.uniform(0.1, 3.0)  # Gravity in Earth G
    atmosphere_composition = {
        "O2": np.random.uniform(0.01, 0.5),  # Oxygen levels (%)
        "CO2": np.random.uniform(0.01, 0.5),  # Carbon dioxide levels (%)
        "Other Gases": np.random.uniform(0.01, 0.9),  # Other gases (%)
    }
    radiation_level = np.random.uniform(1, 500)  # Radiation level (mSv/year)
    temperature_range = (np.random.uniform(-100, 0), np.random.uniform(0, 100))  # Min and max temperature (°C)
    day_length = np.random.uniform(6, 48)  # Day length in hours
    
    return {
        "planet_name": planet_name,
        "star_name": star_name,
        "gravity": gravity,
        "atmosphere_composition": atmosphere_composition,
        "radiation_level": radiation_level,
        "temperature_range": temperature_range,
        "day_length": day_length,
    }

# Objective function to evaluate fitness of a genetic profile
def objective_function(genetic_profile, planet_params):
    """Evaluates the fitness of a genetic profile based on planet conditions."""
    # Extract planet parameters
    gravity = planet_params["gravity"]
    atmosphere = planet_params["atmosphere_composition"]
    radiation = planet_params["radiation_level"]
    temp_min, temp_max = planet_params["temperature_range"]
    day_length = planet_params["day_length"]
    
    # Genetic traits in the profile
    radiation_resistance, bone_density, oxygen_efficiency, temp_adaptability, stress_resilience = genetic_profile
    
    # Fitness components
    fitness_radiation = np.exp(-radiation / radiation_resistance)  # Better resistance reduces impact
    fitness_gravity = np.exp(-abs(gravity - 1) / bone_density)  # Closer to Earth's gravity is ideal
    fitness_oxygen = oxygen_efficiency * atmosphere["O2"]  # Oxygen utilization adapts to O2 levels
    fitness_temperature = np.exp(-abs(temp_min + temp_max) / (2 * temp_adaptability))  # Avg temp adaptation
    fitness_stress = stress_resilience / day_length  # Better stress handling for long days

    # Combined fitness score (weighted sum)
    fitness = (0.25 * fitness_radiation +
               0.2 * fitness_gravity +
               0.25 * fitness_oxygen +
               0.2 * fitness_temperature +
               0.1 * fitness_stress)
    return fitness

# Parameters for Genetic Algorithm
population_size = 150
num_generations = 300
num_genes = 5  # Number of genetic traits
mutation_rate = 0.1

# Initialize population
population = np.random.uniform(0.5, 5.0, size=(population_size, num_genes))  # Random genetic profiles
planet_params = generate_planet()
fitness_history = []

# Optimization loop
for generation in range(num_generations):
    # Evaluate fitness for each individual
    fitness = np.array([objective_function(individual, planet_params) for individual in population])
    fitness_history.append(np.max(fitness))  # Track the best fitness in this generation

    # Print generation progress
    best_individual = population[np.argmax(fitness)]
    print(f"Generation {generation + 1}: Best Fitness = {np.max(fitness):.4f}")

    # Selection (roulette wheel selection)
    probabilities = fitness / fitness.sum()
    selected_indices = np.random.choice(np.arange(population_size), size=population_size, p=probabilities)
    selected_population = population[selected_indices]

    # Crossover (single-point)
    new_population = []
    for i in range(0, population_size, 2):
        parent1, parent2 = selected_population[i], selected_population[(i + 1) % population_size]
        crossover_point = np.random.randint(1, num_genes)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        new_population.extend([child1, child2])

    # Mutation
    new_population = np.array(new_population)
    mutation_mask = np.random.rand(population_size, num_genes) < mutation_rate
    new_population[mutation_mask] += np.random.normal(0, 0.1, size=mutation_mask.sum())

    # Update population
    population = new_population

# Final results
final_fitness = np.array([objective_function(individual, planet_params) for individual in population])
best_individual = population[np.argmax(final_fitness)]
print("\nOptimization Completed!")
print(f"Planet Parameters: {planet_params}")
print(f"Best Genetic Profile: {best_individual}")
print(f"Best Fitness: {np.max(final_fitness):.4f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(fitness_history, label="Best Fitness per Generation")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Genetic Algorithm Optimization for Human Survival on Exoplanet")
plt.legend()
plt.grid()
plt.show()
