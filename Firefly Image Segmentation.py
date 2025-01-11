import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, exposure
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from skimage.color import label2rgb

def cluster_cost(m, X):
    """
    Calculate the cost for clustering.
    """
    d = pairwise_distances(X, m, metric='euclidean')
    dmin = np.min(d, axis=1)
    ind = np.argmin(d, axis=1)
    WCD = np.sum(dmin)
    return WCD, {'d': d, 'dmin': dmin, 'ind': ind, 'WCD': WCD}

# Load image
img = io.imread('f.jpg')
img = img / 255.0  # Normalize to [0, 1]
gray = color.rgb2gray(img)
gray = exposure.equalize_adapthist(gray)

# Reshape image to vector
X = gray.reshape(-1, 1)

# Firefly Algorithm Parameters
k = 10  # Number of clusters
MaxIt = 50  # Maximum Number of Iterations
nPop = 5  # Number of Fireflies
gamma = 1  # Light Absorption Coefficient
beta0 = 2  # Attraction Coefficient Base Value
alpha = 0.2  # Mutation Coefficient
alpha_damp = 0.98  # Mutation Coefficient Damping Ratio
delta = 0.05 * (X.max() - X.min())  # Uniform Mutation Range
m = 2  # Distance exponent

dmax = np.linalg.norm(X.max() - X.min())

# Initialize firefly population
fireflies = [{'Position': np.random.uniform(X.min(), X.max(), (k, 1)), 'Cost': np.inf, 'Out': None} for _ in range(nPop)]

# Evaluate initial population
BestSol = {'Cost': np.inf}
for firefly in fireflies:
    firefly['Cost'], firefly['Out'] = cluster_cost(firefly['Position'], X)
    if firefly['Cost'] < BestSol['Cost']:
        BestSol = firefly.copy()

BestCost = []

# Firefly Algorithm Main Loop
for it in range(MaxIt):
    new_fireflies = []
    for i, firefly_i in enumerate(fireflies):
        new_firefly = {'Cost': np.inf}
        for j, firefly_j in enumerate(fireflies):
            if firefly_j['Cost'] < firefly_i['Cost']:
                rij = np.linalg.norm(firefly_i['Position'] - firefly_j['Position']) / dmax
                beta = beta0 * np.exp(-gamma * rij**m)
                e = delta * np.random.uniform(-1, 1, firefly_i['Position'].shape)
                new_position = firefly_i['Position'] + beta * np.random.rand(*firefly_i['Position'].shape) * (firefly_j['Position'] - firefly_i['Position']) + alpha * e
                new_position = np.clip(new_position, X.min(), X.max())
                cost, out = cluster_cost(new_position, X)
                if cost < new_firefly['Cost']:
                    new_firefly = {'Position': new_position, 'Cost': cost, 'Out': out}
                    if cost < BestSol['Cost']:
                        BestSol = new_firefly.copy()
        new_fireflies.append(new_firefly)

    fireflies = sorted(fireflies + new_fireflies, key=lambda x: x['Cost'])[:nPop]
    BestCost.append(BestSol['Cost'])
    alpha *= alpha_damp
    print(f"Iteration {it + 1}: Best Cost = {BestSol['Cost']}")

# Reshape best solution
FAlbl = BestSol['Out']['ind']
segmented = label2rgb(FAlbl.reshape(gray.shape))

# Plot results
plt.figure()
plt.plot(BestCost, '--k', linewidth=1)
plt.title('FA Train')
plt.xlabel('FA Iteration Number')
plt.ylabel('FA Best Cost Value')
plt.show()

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(segmented)
plt.title('Segmented Image')
plt.show()
