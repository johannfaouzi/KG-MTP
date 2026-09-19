"""A clean, importable, timed wrapper around the original KGMTP pipeline.

`expri_code/run_KGMTP.py` drives `models.KGMTP` exactly as the paper's own experiments
do (weights matrix built once, three branches -- raw, Hilbert transform, first
difference -- sharing one RNG stream, Hydra features scaled with the paper's own
torch-based `SparseScaler`, then a `StandardScaler` + `RidgeClassifierCV` on top), but
as a hardcoded top-level script over all 112 UCR datasets with 30 resamples each,
writing straight to CSV. That's the right shape for reproducing the paper's own
results table, but the wrong shape for anyone who just wants to run this pipeline
programmatically on one dataset and get timing/accuracy/features back -- e.g. to
compare against a reimplementation elsewhere.

`run_kgmtp_pipeline` factors out exactly that: the same weights construction and
three-branch orchestration as `run_KGMTP.py`, on a single train/test split, timed, and
returned as a plain dict instead of printed/written to disk. It changes no modeling
logic -- it's `run_KGMTP.py`'s per-resample body, extracted and made reusable.
"""

import time
from itertools import combinations

import numpy as np
import torch
from scipy import fftpack
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.KGMTP import KGMTP
from utils.tools import SparseScaler

_KERNEL_LENGTH = 6
_GROUP_MULTIPLIERS = {1: 1.0, 2: 1.0, 3: 1.0, 4: 2.0, 5: 5.0}


def build_weights(kernel_length=_KERNEL_LENGTH):
    """Build the fixed kernel-weight matrix, exactly as `run_KGMTP.py` does inline.

    One block per `value_length` (1 through `kernel_length - 1`), each row a kernel
    with `value_length` positive positions and the rest negative, scaled by
    `_GROUP_MULTIPLIERS`. Returns a single `(num_kernels_total, kernel_length)`
    float32 array, in the same row order `run_KGMTP.py` builds `weights1..5` and
    concatenates them.
    """
    blocks = []
    for value_length in range(1, kernel_length):
        indices = np.array(
            list(combinations(np.arange(kernel_length), value_length)), dtype=np.int32
        )
        num_kernels = len(indices)
        block = np.full((num_kernels, kernel_length), -1.0, dtype=np.float32)
        positive_value = (kernel_length - value_length) * 1 / value_length
        for i in range(num_kernels):
            for j in range(value_length):
                block[i, indices[i, j]] = positive_value
        block *= _GROUP_MULTIPLIERS[value_length]
        blocks.append(block)
    return np.concatenate(blocks, axis=0)


def _hilbert_transform_loop(X):
    """Row-by-row `scipy.fftpack.hilbert`, exactly as `run_KGMTP.py` computes it.

    `run_KGMTP.py` builds this as a float32 array, then implicitly upcasts it back to
    float64 via `np.c_[x_train, x_train_hilbert]` before ever calling `KGMTP.fit`/
    `.predict` (both `njit`-compiled for float64 input) -- the float32 rounding is
    real and intentional (it's the paper's own precision choice), but the final dtype
    handed to `KGMTP` must be float64, so that upcast is done here explicitly instead
    of relying on a concatenation to do it implicitly.
    """
    out = np.zeros(X.shape, dtype=np.float32)
    for i in range(X.shape[0]):
        out[i] = fftpack.hilbert(X[i])
    return out.astype(np.float64)


def run_kgmtp_pipeline(x_train, y_train, x_test, y_test, n_kernels=50_000, seed=0):
    """Fit and evaluate the original KGMTP pipeline on one train/test split, timed.

    Reproduces `expri_code/run_KGMTP.py`'s per-resample body exactly (same weights,
    same three-branch RNG-sharing order: base, Hilbert, diff; same Hydra scaling via
    the real torch `SparseScaler`; same `StandardScaler` + `RidgeClassifierCV`
    classifier), but on a single caller-supplied train/test split instead of the 112
    datasets x 30 resamples `run_KGMTP.py` loops over, and returns results instead of
    printing/writing them.

    Parameters
    ----------
    x_train, x_test : 2D float64 array of shape (n_cases, n_timepoints)
        Raw (untransformed) univariate series.
    y_train, y_test : 1D array-like
        Class labels.
    n_kernels : int, default=50_000
        Total feature budget, split evenly across the three branches (matches
        `KGMTP`'s own `num_features` argument, and `run_KGMTP.py`'s `num_features`).
    seed : int, default=0
        Seed for the `numpy.random.Generator` driving all three branches' bias
        fitting, in the same shared-stream order `run_KGMTP.py` uses.

    Returns
    -------
    dict with keys:
        train_fit_transform_runtime : float
            Seconds for fitting all three branches on `x_train` (a cold fit +
            transform of the training data, as `KGMTP.fit` returns both) --
            including each non-base branch's Hilbert-transform/diff preprocessing
            of `x_train`, timed inline rather than upfront, so this reflects the
            real cost of going from raw `x_train` to fitted features.
        test_transform_runtime : float
            Seconds for transforming `x_test` alone (`KGMTP.predict`, post-fit) --
            including each non-base branch's Hilbert-transform/diff preprocessing
            of `x_test`, timed the same way as above.
        accuracy_test : float
            Classification accuracy on `x_test`/`y_test`, using the real pipeline
            (torch `SparseScaler` on Hydra features, `StandardScaler` +
            `RidgeClassifierCV` on the concatenation).
        raw_train, raw_test : 2D float32 array
            The three branches' raw, pre-scaling PPV + Hydra features concatenated
            (`n_kernels // 3` each), before either scaler is applied -- useful for
            comparing against another implementation's own raw features without
            conflating "different math" with "different scaling choices".
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)

    weights = build_weights()
    rng = np.random.default_rng(seed)

    n_kernels_per_branch = n_kernels // 3

    # Hilbert-transform/diff preprocessing is timed *inside* both windows below,
    # rather than upfront -- a caller going from raw x_train/x_test to fitted
    # features (or to transformed test features) pays this cost either way, and
    # excluding it here would understate this implementation's real runtime
    # relative to anything that counts it (e.g. aeon's `KGMTP.fit`/`.transform`,
    # which compute their own Hilbert transform + diff internally).
    t0 = time.perf_counter()
    branch_base = KGMTP(num_features=n_kernels_per_branch, weights=weights)
    train_feat, train_hydra = branch_base.fit(x_train=x_train, rng=rng)

    x_train_hilbert = _hilbert_transform_loop(x_train)
    branch_hilbert = KGMTP(num_features=n_kernels_per_branch, weights=weights)
    train_feat_h, train_hydra_h = branch_hilbert.fit(x_train=x_train_hilbert, rng=rng)

    x_train_diff = np.diff(x_train, 1)
    branch_diff = KGMTP(num_features=n_kernels_per_branch, weights=weights)
    train_feat_d, train_hydra_d = branch_diff.fit(x_train=x_train_diff, rng=rng)
    train_fit_transform_runtime = time.perf_counter() - t0

    t0 = time.perf_counter()
    test_feat, test_hydra = branch_base.predict(x_test)
    x_test_hilbert = _hilbert_transform_loop(x_test)
    test_feat_h, test_hydra_h = branch_hilbert.predict(x_test_hilbert)
    x_test_diff = np.diff(x_test, 1)
    test_feat_d, test_hydra_d = branch_diff.predict(x_test_diff)
    test_transform_runtime = time.perf_counter() - t0

    train_features = np.c_[train_feat, train_feat_h, train_feat_d]
    train_hydra_all = np.c_[train_hydra, train_hydra_h, train_hydra_d]
    test_features = np.c_[test_feat, test_feat_h, test_feat_d]
    test_hydra_all = np.c_[test_hydra, test_hydra_h, test_hydra_d]

    scaler = SparseScaler()
    train_hydra_scaled = np.array(
        scaler.fit_transform(torch.FloatTensor(train_hydra_all))
    )
    test_hydra_scaled = np.array(scaler.transform(torch.FloatTensor(test_hydra_all)))

    classifier = make_pipeline(
        StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    )
    classifier.fit(np.c_[train_features, train_hydra_scaled], y_train)
    predictions = classifier.predict(np.c_[test_features, test_hydra_scaled])
    accuracy_test = float(np.mean(predictions == np.asarray(y_test)))

    return {
        "train_fit_transform_runtime": train_fit_transform_runtime,
        "test_transform_runtime": test_transform_runtime,
        "accuracy_test": accuracy_test,
        "raw_train": np.concatenate([train_features, train_hydra_all], axis=1),
        "raw_test": np.concatenate([test_features, test_hydra_all], axis=1),
    }
