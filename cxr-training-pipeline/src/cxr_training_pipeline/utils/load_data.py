import numpy as np


def make_train_val_indices(train_val_df, val_size=0.25, random_state=42):
    train_val_df = train_val_df.reset_index(drop=True)

    n_samples = len(train_val_df)
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed=random_state)
    rng.shuffle(indices)

    split_idx = int(n_samples * (1 - val_size))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)

    return train_idx, val_idx


def load_mmap_array(path: str, shape: tuple):
    return np.memmap(path, dtype="uint8",mode="r", shape=shape)