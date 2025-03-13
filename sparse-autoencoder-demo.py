# Cell 1: Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Cell 2: Set up the dataset
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
batch_size = 128
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Cell 3: Define the base CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 128 is our latent dimension
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        latent = F.relu(self.fc1(x))  # This is the latent representation
        x = self.fc2(latent)
        return x, latent
    
    def get_latent(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        latent = F.relu(self.fc1(x))
        return latent

# Cell 4: Initialize the model
model = SimpleCNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Cell 5: Train the base model (quick training for demo)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training base model for 1 epoch...")
for i, data in enumerate(tqdm(trainloader)):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    
    optimizer.zero_grad()
    
    outputs, _ = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    if i >= 100:  # Limit training to just a portion of the dataset for demo
        break

print('Base model ready for latent space analysis')

# Cell 6: Evaluate the base model
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
    return accuracy

evaluate_model(model, testloader)

# Cell 7: Collect latent representations
def collect_latent_representations(model, dataloader, n_samples=1000):
    model.eval()
    latent_vectors = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            latent = model.get_latent(inputs)
            
            latent_vectors.append(latent.cpu().numpy())
            labels_list.append(labels.numpy())
            
            if len(labels_list) * batch_size >= n_samples:
                break
    
    latent_vectors = np.vstack(latent_vectors)[:n_samples]
    labels_list = np.concatenate(labels_list)[:n_samples]
    
    return latent_vectors, labels_list

print("Collecting latent representations...")
latent_vectors, labels = collect_latent_representations(model, testloader)
print(f"Collected {latent_vectors.shape[0]} latent vectors of dimension {latent_vectors.shape[1]}")

# Cell 8: Visualize some original latent representations
plt.figure(figsize=(12, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.plot(latent_vectors[i])
    plt.title(f'Digit {labels[i]}')
    if i == 0:
        plt.ylabel('Activation')
    plt.xlabel('Latent Dimension')
plt.tight_layout()
plt.suptitle('Original Latent Representations', y=1.05)
plt.show()

# Cell 9: Define the Sparse Autoencoder
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # Encoder: input_dim -> hidden_dim
        encoded = torch.sigmoid(self.encoder(x))
        # Decoder: hidden_dim -> input_dim
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def get_features(self, x):
        return torch.sigmoid(self.encoder(x))

# Cell 10: Initialize the Sparse Autoencoder
# Hyperparameters for sparsity
sparsity_target = 0.05  # Target activation for hidden neurons
sparsity_weight = 0.1   # Weight of the sparsity penalty term
l1_weight = 0.001       # Weight of L1 regularization

# Define the sparse autoencoder
latent_dim = latent_vectors.shape[1]  # 128 from our CNN
hidden_dim = 256  # Larger than input dim for overcomplete representation
autoencoder = SparseAutoencoder(latent_dim, hidden_dim).to(device)

# Define optimizer
ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Convert latent vectors to torch tensors
latent_tensor = torch.FloatTensor(latent_vectors).to(device)

# Cell 11: Train the Sparse Autoencoder
# Training loop for the sparse autoencoder
print("Training the sparse autoencoder...")
num_epochs = 50
losses = []

for epoch in tqdm(range(num_epochs)):
    # Forward pass
    encoded, decoded = autoencoder(latent_tensor)
    
    # Compute reconstruction loss (MSE)
    mse_loss = F.mse_loss(decoded, latent_tensor)
    
    # Compute sparsity penalty (KL divergence)
    # Average activation of each hidden unit
    rho_hat = torch.mean(encoded, dim=0)
    # Target sparsity
    rho = torch.tensor([sparsity_target] * hidden_dim).to(device)
    # KL divergence
    kl_div = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    sparsity_penalty = torch.sum(kl_div)
    
    # L1 regularization to encourage sparsity
    l1_penalty = torch.sum(torch.abs(encoded))
    
    # Total loss
    loss = mse_loss + sparsity_weight * sparsity_penalty + l1_weight * l1_penalty
    losses.append([epoch, loss.item(), mse_loss.item(), sparsity_penalty.item(), l1_penalty.item()])
    
    # Backward pass and optimization
    ae_optimizer.zero_grad()
    loss.backward()
    ae_optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, MSE: {mse_loss.item():.4f}, Sparsity: {sparsity_penalty.item():.4f}, L1: {l1_penalty.item():.4f}')

print("Sparse autoencoder training complete!")

# Cell 12: Plot training loss
losses = np.array(losses)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses[:, 0], losses[:, 1], label='Total Loss')
plt.plot(losses[:, 0], losses[:, 2], label='MSE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(losses[:, 0], losses[:, 3], label='Sparsity Penalty')
plt.plot(losses[:, 0], losses[:, 4], label='L1 Penalty')
plt.xlabel('Epoch')
plt.ylabel('Penalty')
plt.legend()
plt.tight_layout()
plt.show()

# Cell 13: Get encoded features and analyze sparsity
# Get the encoded features for our data
autoencoder.eval()
with torch.no_grad():
    encoded_features, _ = autoencoder(latent_tensor)
    encoded_features = encoded_features.cpu().numpy()

# Calculate activation statistics
feature_means = np.mean(encoded_features, axis=0)
feature_activations = np.mean(encoded_features > 0.5, axis=0)

# Cell 14: Plot activation distribution
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(feature_means, bins=50)
plt.title('Distribution of Feature Activation Means')
plt.xlabel('Mean Activation')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist(feature_activations, bins=50)
plt.title('Percentage of Samples Activating Each Feature')
plt.xlabel('Activation Percentage')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Cell 15: Visualize feature sparsity
# Calculate the sparsity of activations
sparsity = np.mean(encoded_features < 0.1)
print(f"Percentage of near-zero activations: {sparsity * 100:.2f}%")

# Calculate average number of active features per sample
active_features_per_sample = np.sum(encoded_features > 0.5, axis=1)
avg_active_features = np.mean(active_features_per_sample)
print(f"Average number of active features per sample: {avg_active_features:.2f} out of {hidden_dim}")

# Plot distribution of active features per sample
plt.figure(figsize=(10, 5))
plt.hist(active_features_per_sample, bins=50)
plt.title('Distribution of Active Features Per Sample')
plt.xlabel('Number of Active Features')
plt.ylabel('Number of Samples')
plt.axvline(avg_active_features, color='r', linestyle='--', 
           label=f'Mean: {avg_active_features:.2f}')
plt.legend()
plt.show()

# Cell 16: Apply t-SNE to visualize latent spaces
# Apply t-SNE to original latent space
print("Applying t-SNE to original latent vectors...")
tsne_original = TSNE(n_components=2, random_state=42)
tsne_results_original = tsne_original.fit_transform(latent_vectors)

# Apply t-SNE to sparse autoencoder features
print("Applying t-SNE to sparse autoencoder features...")
tsne_sparse = TSNE(n_components=2, random_state=42)
tsne_results_sparse = tsne_sparse.fit_transform(encoded_features)

# Cell 17: Plot t-SNE visualizations
# Plot the t-SNE results
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
scatter = plt.scatter(tsne_results_original[:, 0], tsne_results_original[:, 1], 
                      c=labels, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label='Digit Class')
plt.title('t-SNE of Original Latent Space')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

plt.subplot(1, 2, 2)
scatter = plt.scatter(tsne_results_sparse[:, 0], tsne_results_sparse[:, 1], 
                      c=labels, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label='Digit Class')
plt.title('t-SNE of Sparse Autoencoder Features')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

plt.tight_layout()
plt.show()

# Cell 18: Visualize what features are detecting
# Get weights from the decoder part of the autoencoder
decoder_weights = autoencoder.decoder.weight.data.cpu().numpy()

# Find the top activated features across the dataset
mean_activations = np.mean(encoded_features, axis=0)
top_features_idx = np.argsort(mean_activations)[-10:]  # Top 10 most activated features

# FIXED: Calculate the correct shape for visualization
# Each row in decoder_weights is the weights from one hidden unit to all latent dimensions
# Each hidden unit connects to all 128 latent dimensions
print(f"Decoder weight matrix shape: {decoder_weights.shape}")
print(f"Latent dimension: {latent_dim}")

# Plot the decoder weights for the top features as a 1D plot
plt.figure(figsize=(15, 8))
for i, idx in enumerate(top_features_idx):
    plt.subplot(2, 5, i+1)
    plt.plot(decoder_weights.T[idx])
    plt.title(f'Feature {idx}\nMean Act: {mean_activations[idx]:.4f}')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Weight')
plt.tight_layout()
plt.suptitle('Top 10 Most Activated Features (Decoder Weights)', y=1.05)
plt.show()

# Cell 19: Compare original and reconstructed representations
# Reconstruct the latent vectors using the autoencoder
with torch.no_grad():
    _, reconstructed = autoencoder(latent_tensor)
    reconstructed = reconstructed.cpu().numpy()

# Calculate reconstruction error
mse = np.mean((latent_vectors - reconstructed) ** 2)
print(f"Mean Squared Error of Reconstruction: {mse:.6f}")

# Plot some example reconstructions
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.plot(latent_vectors[i])
    plt.title(f'Original (Digit {labels[i]})')
    plt.ylim(0, np.max(latent_vectors) * 1.1)
    if i == 0:
        plt.ylabel('Activation')
    
    plt.subplot(2, 5, i+6)
    plt.plot(reconstructed[i])
    plt.title('Reconstructed')
    plt.xlabel('Latent Dimension')
    plt.ylim(0, np.max(latent_vectors) * 1.1)
    if i == 0:
        plt.ylabel('Activation')
plt.tight_layout()
plt.show()

# Cell 20: Analyze feature activation by digit class
# Analyze which features activate most strongly for each digit class
class_feature_activations = []
for digit in range(10):
    mask = labels == digit
    class_activations = encoded_features[mask]
    class_feature_activations.append(np.mean(class_activations, axis=0))

# Find distinctive features for each class
distinctive_features = {}
for digit in range(10):
    # Compare this class's feature activations to the average of all other classes
    other_classes_avg = np.mean([class_feature_activations[i] for i in range(10) if i != digit], axis=0)
    difference = class_feature_activations[digit] - other_classes_avg
    top_distinctive = np.argsort(difference)[-5:]  # Top 5 most distinctive features
    distinctive_features[digit] = top_distinctive

# Cell 21: Visualize class-specific features
# Plot feature activation patterns by class
plt.figure(figsize=(15, 10))
for digit in range(10):
    plt.subplot(2, 5, digit+1)
    plt.bar(range(5), [class_feature_activations[digit][idx] for idx in distinctive_features[digit]])
    plt.title(f'Digit {digit}')
    plt.xlabel('Distinctive Feature Index')
    plt.ylabel('Mean Activation')
    plt.xticks(range(5), [distinctive_features[digit][i] for i in range(5)])
plt.tight_layout()
plt.suptitle('Most Distinctive Features for Each Digit Class', y=1.02, fontsize=16)
plt.show()

# Cell 22: Advanced analysis - Feature contribution heatmap
plt.figure(figsize=(12, 8))
feature_importance = np.zeros((10, 10))

for i, digit in enumerate(range(10)):
    for j, feature_idx in enumerate(distinctive_features[digit][:10]):
        if j < 10:  # Only use top 10 features
            feature_importance[i, j] = class_feature_activations[digit][feature_idx]

plt.imshow(feature_importance, cmap='viridis')
plt.colorbar(label='Mean Activation')
plt.xlabel('Top Distinctive Feature Index')
plt.ylabel('Digit Class')
plt.title('Feature Importance Heatmap by Digit Class')
plt.xticks(range(10))
plt.yticks(range(10))
plt.show()

# Cell 23: Alternative visualization - weight patterns
# Get encoder weights for more interpretable visualization
encoder_weights = autoencoder.encoder.weight.data.cpu().numpy()

# Attempt to visualize feature detectors by reshaping if possible
plt.figure(figsize=(15, 8))
plt.suptitle('Alternative Feature Visualization - Encoder Weights', y=1.05)

# For interpretability, let's visualize the encoder weights that connect to most impactful latent dims
for i, idx in enumerate(top_features_idx[:10]):  # Top 10 features
    feature_weights = encoder_weights[idx]
    
    # Find most strongly connected latent dimensions for this feature
    strongest_connections = np.argsort(np.abs(feature_weights))[-16:]  # Top 16 connections
    
    plt.subplot(2, 5, i+1)
    plt.stem(strongest_connections, feature_weights[strongest_connections])
    plt.title(f'Feature {idx} Connections')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Weight')
plt.tight_layout()
plt.show()

# Cell 24: Conclusion and summary
print("""
# Conclusion and Summary:

In this notebook, we've demonstrated how to use sparse autoencoders to analyze the latent space of a deep neural network. Key findings include:

1. **Sparsity Properties**:
   - The sparse autoencoder learned a representation where only about {:.1f} features are active per sample on average.
   - Approximately {:.1f}% of all feature activations are near zero.

2. **Interpretability**:
   - The sparse representation maintains class separation as shown in the t-SNE visualization.
   - Different digits activate distinct sets of features.
   - Individual features correspond to specific patterns in the original latent space.

3. **Reconstruction Quality**:
   - The autoencoder successfully reconstructs the original latent representations with an MSE of {:.6f}.
   - The information preserved in the sparse code is sufficient to retain class-specific information.

4. **Feature Analysis**:
   - We identified the most distinctive features for each digit class.
   - The visualization of decoder weights shows the connection patterns between sparse features and latent dimensions.

These techniques can be applied to analyze and interpret the latent spaces of various deep learning models, providing insights into what the models have learned and how they represent data internally.
""".format(avg_active_features, sparsity * 100, mse))

# Cell 25: Bonus - Representative samples for each feature
# Find representative examples that most strongly activate each feature
def find_representative_samples(encoded_features, latent_vectors, labels, feature_idx, top_k=5):
    # Sort samples by activation of the given feature
    sorted_indices = np.argsort(encoded_features[:, feature_idx])[::-1]
    top_indices = sorted_indices[:top_k]
    
    return top_indices, encoded_features[top_indices, feature_idx]

# Choose a feature to analyze
feature_to_analyze = top_features_idx[-1]  # Most activated feature

# Find samples that strongly activate this feature
top_sample_indices, activation_values = find_representative_samples(
    encoded_features, latent_vectors, labels, feature_to_analyze)

plt.figure(figsize=(12, 4))
plt.suptitle(f'Representative Samples for Feature {feature_to_analyze}', y=1.05)

for i, idx in enumerate(top_sample_indices):
    plt.subplot(1, 5, i+1)
    plt.plot(latent_vectors[idx])
    plt.title(f'Digit {labels[idx]}\nAct: {activation_values[i]:.3f}')
    plt.xlabel('Latent Dimension')
    if i == 0:
        plt.ylabel('Activation')
plt.tight_layout()
plt.show()
