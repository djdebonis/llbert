import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEED = 1992


def split_indices(data, test_size=DEFAULT_TEST_SIZE, seed=DEFAULT_SEED):
    """Return (train_indices, val_indices) arrays.

    Uses a group-aware split on the ``intersection`` column when present so
    that every row from the same intersection lands in the same partition.
    Falls back to a plain random split otherwise.
    """
    indices = np.arange(len(data))
    if "intersection" in data.columns:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )
        train_idx, val_idx = next(
            splitter.split(indices, groups=data["intersection"])
        )
    else:
        train_idx, val_idx = train_test_split(
            indices, test_size=test_size, random_state=seed
        )
    return train_idx, val_idx
