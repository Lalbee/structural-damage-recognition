import numpy as np

MEAN = np.array([123.68, 116.779, 103.939], dtype=np.float32)

# adjust these paths to your folders
FILES = {
    "train": ("DATASET/task2_damage_state_2/task2_X_train.npy",
              "DATASET/task2_damage_state_1/task2_y_train.npy"),
    "test":  ("DATASET/task2_damage_state_1/task2_X_test.npy",
              "DATASET/task2_damage_state_1/task2_y_test.npy"),
}

for split, (x_path, y_path) in FILES.items():
    X = np.load(x_path, mmap_mode="r")        # reads from disk in chunks
    y = np.load(y_path).argmax(axis=1).astype(np.int64)

    out = np.empty(X.shape, dtype=np.uint8)
    for i in range(0, len(X), 512):
        out[i:i + 512] = (X[i:i + 512] + MEAN).clip(0, 255).astype(np.uint8)

    np.save(f"data/X_{split}.npy", out)
    np.save(f"data/y_{split}.npy", y)
    print(split, out.shape, np.bincount(y))