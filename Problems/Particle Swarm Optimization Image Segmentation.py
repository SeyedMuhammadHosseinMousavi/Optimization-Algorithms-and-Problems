import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util
from sklearn.metrics import pairwise_distances
from scipy.ndimage import median_filter

# Cost Function for PSO-SA
def clu_cos_pso_sa(m, X, clusters):
    g = m.reshape(clusters, 3)
    d = pairwise_distances(X, g, metric='euclidean')
    dmin = np.min(d, axis=1)
    WCD = np.sum(dmin)
    return WCD

# Initialize PSO Parameters
class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position
        self.best_cost = np.inf
        self.cost = np.inf

# PSO Parameters
swarm_size = 20
iterations = 20
w = 0.7  # Inertia weight
c1 = 1.5  # Personal acceleration coefficient
c2 = 1.5  # Global acceleration coefficient

# Simulated Annealing Parameters
sa_iterations = 100
sa_temp = 100
sa_cooling = 0.7

# Load Image
main_org = io.imread('tst.jpg')
gray = color.rgb2gray(main_org)

# Normalize RGB channels
R, G, B = main_org[:, :, 0], main_org[:, :, 1], main_org[:, :, 2]
X1 = (R - np.min(R)) / (np.max(R) - np.min(R))
X2 = (G - np.min(G)) / (np.max(G) - np.min(G))
X3 = (B - np.min(B)) / (np.max(B) - np.min(B))
X = np.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=1)

# Clustering Parameters
clusters = 7
VarSize = (clusters, 3)
nVar = np.prod(VarSize)
VarMin = np.min(X, axis=0).repeat(clusters)
VarMax = np.max(X, axis=0).repeat(clusters)

# Initialize Swarm
particles = []
for _ in range(swarm_size):
    position = np.random.uniform(VarMin, VarMax, nVar)
    velocity = np.random.uniform(-1, 1, nVar)
    particle = Particle(position, velocity)
    particle.cost = clu_cos_pso_sa(particle.position, X, clusters)
    particle.best_position = particle.position
    particle.best_cost = particle.cost
    particles.append(particle)

global_best_position = particles[0].best_position
global_best_cost = particles[0].best_cost

for particle in particles:
    if particle.best_cost < global_best_cost:
        global_best_position = particle.best_position
        global_best_cost = particle.best_cost

# PSO Loop
pso_costs = []
for it in range(iterations):
    for particle in particles:
        # Update velocity
        r1 = np.random.rand(nVar)
        r2 = np.random.rand(nVar)
        particle.velocity = (
            w * particle.velocity
            + c1 * r1 * (particle.best_position - particle.position)
            + c2 * r2 * (global_best_position - particle.position)
        )

        # Update position
        particle.position += particle.velocity
        particle.position = np.clip(particle.position, VarMin, VarMax)

        # Evaluate cost
        particle.cost = clu_cos_pso_sa(particle.position, X, clusters)

        # Update personal best
        if particle.cost < particle.best_cost:
            particle.best_position = particle.position
            particle.best_cost = particle.cost

        # Update global best
        if particle.best_cost < global_best_cost:
            global_best_position = particle.best_position
            global_best_cost = particle.best_cost

    pso_costs.append(global_best_cost)
    print(f"Iteration {it + 1}/{iterations}, PSO Best Cost: {global_best_cost}")

# SA Refinement
sa_costs = []
current_position = global_best_position
current_cost = global_best_cost
temp = sa_temp

for it in range(sa_iterations):
    new_position = current_position + np.random.uniform(-0.1, 0.1, nVar)
    new_position = np.clip(new_position, VarMin, VarMax)
    new_cost = clu_cos_pso_sa(new_position, X, clusters)

    if new_cost < current_cost or np.random.rand() < np.exp((current_cost - new_cost) / temp):
        current_position = new_position
        current_cost = new_cost

    sa_costs.append(current_cost)
    temp *= sa_cooling
    print(f"SA Iteration {it + 1}/{sa_iterations}, Cost: {current_cost}")

# Final Results
g = current_position.reshape(clusters, 3)
d = pairwise_distances(X, g, metric='euclidean')
dmin = np.min(d, axis=1)
ind = np.argmin(d, axis=1)

# Segmentation
SA_segmented = ind.reshape(main_org.shape[:2])
color_seg = util.img_as_ubyte(color.label2rgb(SA_segmented, gray))

# Median Filtering
med_gray = median_filter(SA_segmented, size=(5, 5))
med_color1 = median_filter(color_seg[:, :, 0], size=(4, 6))
med_color2 = median_filter(color_seg[:, :, 1], size=(4, 6))
med_color3 = median_filter(color_seg[:, :, 2], size=(4, 6))
med_rgb = np.stack([med_color1, med_color2, med_color3], axis=-1)

# Plot Results
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(main_org)
plt.title('Original')
plt.subplot(2, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title('Gray')
plt.subplot(2, 3, 3)
plt.imshow(SA_segmented, cmap='gray')
plt.title(f'PSO-SA Gray Segmented, Clusters = {clusters}')
plt.subplot(2, 3, 4)
plt.imshow(color_seg)
plt.title(f'PSO-SA Color Segmented, Clusters = {clusters}')
plt.subplot(2, 3, 5)
plt.imshow(med_gray, cmap='gray')
plt.title('PSO-SA Gray Median Filtered')
plt.subplot(2, 3, 6)
plt.imshow(med_rgb)
plt.title('PSO-SA Color Median Filtered')
plt.show()

# Plot PSO and SA Iteration Costs
plt.figure(figsize=(15, 5))

# PSO Plot
plt.subplot(1, 2, 1)
plt.plot(range(1, len(pso_costs) + 1), pso_costs, marker='o', label='PSO Best Cost')
plt.title('PSO Iteration Costs')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.grid()
plt.legend()

# SA Plot
plt.subplot(1, 2, 2)
plt.plot(range(1, len(sa_costs) + 1), sa_costs, marker='x', color='r', label='SA Best Cost')
plt.title('SA Iteration Costs')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
