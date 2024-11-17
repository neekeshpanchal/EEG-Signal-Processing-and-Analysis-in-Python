import numpy as np
from scipy.signal import butter, lfilter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# EEG data generation function
def generate_eeg_data(samples=1000, electrodes=10):
    freq = np.random.uniform(0.5, 30, electrodes)  # Random frequencies
    time = np.linspace(0, 1, samples)
    eeg_data = np.array([np.sin(2 * np.pi * f * time) + 0.5 * np.random.randn(samples) for f in freq])
    return eeg_data, time


# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = lfilter(b, a, data, axis=1)
    return filtered_data


# Visualization function
def visualize_data(time, raw_data, filtered_data, clusters, pca_results):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Plot raw EEG data
    axes[0].set_title("Raw EEG Data")
    for i in range(raw_data.shape[0]):
        axes[0].plot(time, raw_data[i] + i * 2, label=f"Electrode {i+1}")
    axes[0].legend(loc="upper right")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    # Plot filtered EEG data
    axes[1].set_title("Filtered EEG Data")
    for i in range(filtered_data.shape[0]):
        axes[1].plot(time, filtered_data[i] + i * 2, label=f"Electrode {i+1}")
    axes[1].legend(loc="upper right")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")

    # Plot PCA-clustered data
    axes[2].set_title("PCA of EEG Clusters")
    for cluster in np.unique(clusters):
        indices = np.where(clusters == cluster)
        axes[2].scatter(
            pca_results[indices, 0],
            pca_results[indices, 1],
            label=f"Cluster {cluster}",
            alpha=0.7
        )
    axes[2].legend()
    axes[2].set_xlabel("PCA Component 1")
    axes[2].set_ylabel("PCA Component 2")

    plt.tight_layout()
    plt.show()


def main():
    # Parameters
    samples = int(input("Enter the number of samples (e.g., 1000): "))
    electrodes = 10
    lowcut = float(input("Enter the low cutoff frequency (e.g., 0.5): "))
    highcut = float(input("Enter the high cutoff frequency (e.g., 30.0): "))
    fs = float(input("Enter the sampling frequency (e.g., 250.0): "))
    n_clusters = int(input("Enter the number of clusters (e.g., 3): "))

    # Generate synthetic EEG data
    print("Generating synthetic EEG data...")
    eeg_data, time = generate_eeg_data(samples=samples, electrodes=electrodes)

    # Apply bandpass filter
    print("Filtering EEG data...")
    filtered_data = bandpass_filter(eeg_data, lowcut, highcut, fs)

    # Perform KMeans clustering
    print("Clustering EEG data...")
    reshaped_data = filtered_data.reshape(filtered_data.shape[0], -1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reshaped_data)

    # Perform PCA
    print("Applying PCA for visualization...")
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(reshaped_data)

    # Visualize results
    print("Visualizing results...")
    visualize_data(time, eeg_data, filtered_data, clusters, pca_results)


if __name__ == "__main__":
    main()
