"""Utilities for mapping numpy arrays into Pyomo parameter dictionaries."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import product

import numpy as np


def array_to_indexed_dict(
    values: np.ndarray,
    index_sets: Sequence[Iterable[int]] | None = None,
) -> dict[int | tuple[int, int], float]:
    """Convert a 1D or 2D numpy array into a Pyomo-friendly dictionary.

    Parameters
    ----------
    values:
        Array of numeric values that will populate a Pyomo ``Param``.
    index_sets:
        Optional sequence describing the indices for each array dimension. If
        omitted, ``range`` iterators matching the array shape are used.

    Returns
    -------
    dict
        Mapping from index tuples to float values. For one-dimensional arrays the
        keys are the first-dimension indices instead of single-element tuples.
    """

    arr = np.asarray(values, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError("Only 1D or 2D arrays are supported")

    if index_sets is None:
        index_sets = tuple(range(dim) for dim in arr.shape)
    else:
        if len(index_sets) != arr.ndim:
            raise ValueError("Number of index sets must match array dimensions")
        index_sets = tuple(tuple(seq) for seq in index_sets)
        for dim, seq in enumerate(index_sets):
            if len(seq) != arr.shape[dim]:
                raise ValueError("Index set length must match array dimension")

    if arr.ndim == 1:
        indices = index_sets[0]
        return {indices[i]: float(arr[i]) for i in range(arr.shape[0])}

    first, second = index_sets
    return {
        (first[i], second[j]): float(arr[i, j])
        for i, j in product(range(arr.shape[0]), range(arr.shape[1]))
    }
