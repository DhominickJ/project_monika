import matplotlib.pyplot as plt
import numpy as np

def plot_euclidean_distances(reference_point, points):
    # Calculate Euclidean distances
    distances = np.sqrt(np.sum((points - reference_point)**2, axis=1))

    # Create a scatter plot
    plt.scatter(points[:, 0], points[:, 1], c=distances, cmap='viridis')
    plt.colorbar(label='Euclidean Distance')
    
    # Mark the reference point
    plt.scatter(*reference_point, color='red')
    plt.text(*reference_point, 'Reference Point', ha='right')

    plt.show()

# Example usage:
reference_point = np.array([0, 0])
points = np.random.rand(100, 2) * 10 - 5  # Random points between -5 and 5
plot_euclidean_distances(reference_point, points)
