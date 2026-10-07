import numpy as np
from pathlib import Path

# Original prepared training data
X = np.load("data/X_train.npy", mmap_mode="r")
y = np.load("data/y_train.npy")

# Number of images to take from each class
N_PER_CLASS = 2000

# Reproducible random selection
rng = np.random.default_rng(42)

# Find indices for each class
damaged_idx = np.where(y == 0)[0]
undamaged_idx = np.where(y == 1)[0]

# Randomly select 2000 from each class
selected_damaged = rng.choice(damaged_idx, N_PER_CLASS, replace=False)
selected_undamaged = rng.choice(undamaged_idx, N_PER_CLASS, replace=False)

# Combine and shuffle
selected_idx = np.concatenate([
    selected_damaged,
    selected_undamaged
])
rng.shuffle(selected_idx)

# Create output directory
out_dir = Path("data_small")
out_dir.mkdir(exist_ok=True)

# Create the smaller arrays
X_small = X[selected_idx]
y_small = y[selected_idx]

# Save
np.save(out_dir / "X_train.npy", X_small)
np.save(out_dir / "y_train.npy", y_small)

print("Small dataset created!")
print("X shape:", X_small.shape)
print("y shape:", y_small.shape)
print("Class counts:", np.bincount(y_small))
print("dtype:", X_small.dtype)