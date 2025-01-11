import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.6, random_state=42)

# Build a Variational Autoencoder (VAE)
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(latent_dim * 2)  # Mean and LogVar
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(X_train.shape[1])
        ])

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def call(self, inputs):
        x = self.encoder(inputs)
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mean, logvar

# Define VAE loss
def vae_loss(data, reconstructed, mean, logvar):
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(data, reconstructed))
    kl_divergence = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar))
    return reconstruction_loss + kl_divergence

# Train VAE
latent_dim = 2
vae = VAE(latent_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

@tf.function
def train_step(data):
    with tf.GradientTape() as tape:
        reconstructed, mean, logvar = vae(data)
        loss = vae_loss(data, reconstructed, mean, logvar)
    gradients = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    return loss

print("Training VAE...")
for epoch in range(200):
    loss = train_step(X_train)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.numpy():.4f}")

# Gray Wolf Optimizer (GWO)
class GrayWolfOptimizer:
    def __init__(self, latent_dim, n_wolves=30, max_iters=200):
        self.latent_dim = latent_dim
        self.n_wolves = n_wolves
        self.max_iters = max_iters
        self.wolves = np.random.uniform(-2, 2, size=(n_wolves, latent_dim))

    def fitness(self, wolves):
        synthetic_data = vae.decoder(tf.convert_to_tensor(wolves, dtype=tf.float32)).numpy()
        reconstruction_loss = np.mean((synthetic_data - np.mean(X_train, axis=0))**2)
        diversity_score = np.mean(np.std(synthetic_data, axis=0))
        return -reconstruction_loss + diversity_score  # Maximize diversity, minimize reconstruction error

    def optimize(self):
        for t in range(self.max_iters):
            fitness = self.fitness(self.wolves)
            sorted_indices = np.argsort(fitness)[::-1]
            self.wolves = self.wolves[sorted_indices]

            # Handle edge cases for population size
            if len(self.wolves) < 3:
                alpha = beta = delta = self.wolves[0]
            else:
                alpha, beta, delta = self.wolves[:3]

            for i in range(len(self.wolves)):
                a = 2 - t * (2 / self.max_iters)
                r1, r2 = np.random.rand(), np.random.rand()
                A1, A2, A3 = 2 * a * r1 - a, 2 * a * r2 - a, 2 * a * np.random.rand() - a
                D1, D2, D3 = abs(A1 * alpha - self.wolves[i]), abs(A2 * beta - self.wolves[i]), abs(A3 * delta - self.wolves[i])
                X1, X2, X3 = alpha - A1 * D1, beta - A2 * D2, delta - A3 * D3
                self.wolves[i] = (X1 + X2 + X3) / 3
        return self.wolves[:min(len(self.wolves), 200)]  # Return top 50 latent vectors

# Generate synthetic data
print("Optimizing latent space with GWO...")
gwo = GrayWolfOptimizer(latent_dim=latent_dim)
optimized_latents = gwo.optimize()
synthetic_data = vae.decoder(tf.convert_to_tensor(optimized_latents, dtype=tf.float32)).numpy()


# Combine original and synthetic data
combined_X_train = np.vstack([X_train, synthetic_data])
synthetic_labels = np.tile(np.argmax(y_train[:len(synthetic_data)], axis=1), (len(synthetic_data) // len(y_train) + 1))[:len(synthetic_data)]
combined_y_train = np.hstack([np.argmax(y_train, axis=1), synthetic_labels])

# Train classifier on combined data
clf_combined = RandomForestClassifier(random_state=42)
clf_combined.fit(combined_X_train, combined_y_train)

# Evaluate on test data
y_combined_pred = clf_combined.predict(X_test)

# Print classification report
print("\nClassification Report (Combined Original and Synthetic Data):")
print(classification_report(np.argmax(y_test, axis=1), y_combined_pred))
