import numpy as np
import pandas as pd

# Example: Additive Noise
def add_noise(data, noise_level=0.01):
    noise = noise_level * np.random.randn(*data.shape)
    return data + noise

# Example: Scaling
def scale_data(data, scale_factor=1.1):
    return data * scale_factor

# Example: Jittering
def jitter_data(data, jitter_level=0.01):
    jitter = jitter_level * np.random.randn(*data.shape)
    return data + jitter

# Select a subset of the data for augmentation (10% of the dataset)
def augment_dataset(df, target_col, augmentation_functions, fraction=0.1):
    augmented_df = df.copy()
    num_samples = int(len(df) * fraction)
    
    # Randomly select indices to augment
    indices_to_augment = np.random.choice(df.index, size=num_samples, replace=False)
    
    # Apply augmentation functions
    for func in augmentation_functions:
        augmented_df.loc[indices_to_augment, target_col] = func(df.loc[indices_to_augment, target_col])
    
    return augmented_df



def reduce_data(df, fraction=0.1):
    """Remove a fraction of the data randomly."""
    reduced_df = df.copy()
    num_samples_to_remove = int(len(df) * fraction)
    
    # Randomly select indices to remove
    indices_to_remove = np.random.choice(df.index, size=num_samples_to_remove, replace=False)
    
    # Drop the selected rows
    reduced_df.drop(indices_to_remove, inplace=True)
    
    return reduced_df