import warnings

import numpy as np
from scipy import linalg, sparse
from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip, safe_sparse_dot


def randomized_range_finder(
    F, B, *, size, n_iter, power_iteration_normalizer="auto", random_state=None
):
    random_state = check_random_state(random_state)

    # Generating normal random vectors with shape: (A.shape[1], size)
    G = random_state.normal(size=(B.shape[1], size))
    if B.dtype.kind == "f":
        # Ensure f32 is preserved as f32
        G = G.astype(B.dtype, copy=False)

    # Deal with "auto" mode
    # if power_iteration_normalizer == "auto":
    #     if n_iter <= 2:
    #         power_iteration_normalizer = "none"
    #     else:
    #         power_iteration_normalizer = "LU"

    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q
    for i in range(n_iter):
        # if power_iteration_normalizer == "none":
        #     Q = safe_sparse_dot(A, Q)
        #     Q = safe_sparse_dot(A.T, Q)
        # elif power_iteration_normalizer == "LU":
        #     Q, _ = linalg.lu(safe_sparse_dot(A, Q), permute_l=True)
        #     Q, _ = linalg.lu(safe_sparse_dot(A.T, Q), permute_l=True)
        G, _ = linalg.lu(safe_sparse_dot(F, safe_sparse_dot(B, G)), permute_l=True)
        G, _ = linalg.lu(safe_sparse_dot(B.T, safe_sparse_dot(F.T, G)), permute_l=True)
        # elif power_iteration_normalizer == "QR":
        # G, _ = linalg.qr(safe_sparse_dot(F, safe_sparse_dot(B, G)), mode="economic")
        # G, _ = linalg.qr(safe_sparse_dot(B.T, safe_sparse_dot(F.T, G)), mode="economic")

    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    # G, _ = linalg.qr(safe_sparse_dot(A, G), mode="economic")
    G, _ = linalg.qr(safe_sparse_dot(F, safe_sparse_dot(B, G)), mode="economic")
    return G


def randomized_svd(
    F, B,
    n_components,
    *,
    n_oversamples=10,
    n_iter="auto",
    power_iteration_normalizer="auto",
    transpose=False,
    flip_sign=True,
    random_state="warn",
):
    if isinstance(F, (sparse.lil_matrix, sparse.dok_matrix)):
        warnings.warn(
            "Calculating SVD of a {} is expensive. "
            "csr_matrix is more efficient.".format(type(F).__name__),
            sparse.SparseEfficiencyWarning,
        )

    if isinstance(B, (sparse.lil_matrix, sparse.dok_matrix)):
        warnings.warn(
            "Calculating SVD of a {} is expensive. "
            "csr_matrix is more efficient.".format(type(B).__name__),
            sparse.SparseEfficiencyWarning,
        )

    if random_state == "warn":
        warnings.warn(
            "If 'random_state' is not supplied, the current default "
            "is to use 0 as a fixed seed. This will change to  "
            "None in version 1.2 leading to non-deterministic results "
            "that better reflect nature of the randomized_svd solver. "
            "If you want to silence this warning, set 'random_state' "
            "to an integer seed or to None explicitly depending "
            "if you want your code to be deterministic or not.",
            FutureWarning,
        )
        random_state = 0

    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    # n_samples, n_features = F.shape[0], B.shape[1]

    if n_iter == "auto":
        # Checks if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA. See #5299
        n_iter = 7 if n_components < 0.1 * min(B.shape) else 4

    # if transpose == "auto":
    #     transpose = n_samples < n_features
    # if transpose:
    #     # this implementation is a bit faster with smaller shape[1]
    #     M = M.T

    Q = randomized_range_finder(
        F, B,
        size=n_random,
        n_iter=n_iter,
        power_iteration_normalizer=power_iteration_normalizer,
        random_state=random_state,
    )

    B = safe_sparse_dot(safe_sparse_dot(Q.T, F), B)

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, Vt = linalg.svd(B, full_matrices=False)

    del B
    U = np.dot(Q, Uhat)

    if flip_sign:
        if not transpose:
            U, Vt = svd_flip(U, Vt)
        else:
            # In case of transpose u_based_decision=false
            # to actually flip based on u and not v.
            U, Vt = svd_flip(U, Vt, u_based_decision=False)

    if transpose:
        # transpose back the results according to the input convention
        return Vt[:n_components, :].T, s[:n_components], U[:, :n_components].T
    else:
        return U[:, :n_components], s[:n_components], Vt[:n_components, :]