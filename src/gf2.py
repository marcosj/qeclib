import numpy as np

def rref(A: np.ndarray) -> tuple[np.ndarray, list[int]]:
    r'''
    Gaussian elimination over $\mathbb{F}_2$.

    Args:
        A: Binary matrix of shape `(m, n)`.

    Returns:
        `(R, pivots)` where `R` is the reduced row echelon basis and `pivots` are pivot column indices.
    '''
    M = (A.copy() % 2).astype(np.uint8)
    m, n = M.shape
    pivots = []
    r = 0
    for c in range(n):
        if r >= m:
            break
        piv = next((i for i in range(r, m) if M[i, c]), None)
        if piv is None:
            continue
        if piv != r:
            M[[r, piv]] = M[[piv, r]]
        for i in range(m):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        pivots.append(c)
        r += 1
    return M[:r], pivots

def null(A: np.ndarray) -> np.ndarray:
    '''Returns the nullspace of A: all operators that commute with the basis of A'''
    n = A.shape[1]
    R, pivots = rref(A)
    piv_set = set(pivots)
    free_cols = [c for c in range(n) if c not in piv_set]
    basis = np.zeros((len(free_cols), n), dtype=np.uint8)
    for k, f in enumerate(free_cols):
        basis[k, f] = 1
        for i, p in enumerate(pivots):
            basis[k, p] = R[i, f]
    return basis

def inv(A: np.ndarray) -> np.ndarray:
    r'''Finds $A^{−1}\mod 2$ via Gauss-Jordan elimination'''
    k = A.shape[0]
    aug = np.hstack([A.copy() % 2, np.eye(k, dtype=np.uint8)])
    for c in range(k):
        piv = next((r for r in range(c, k) if aug[r, c]), None)
        if piv is None:
            raise ValueError('Matrix is singular over GF(2)')
        if piv != c:
            aug[[c, piv]] = aug[[piv, c]]
        for r in range(k):
            if r != c and aug[r, c]:
                aug[r] ^= aug[c]
    return aug[:, k:]

def quot(ker_of: np.ndarray, mod_by: np.ndarray) -> np.ndarray:
    r'''Computes $L_Z = \ker(H_X) / \operatorname{row}(H_Z) and L_X = \ker(H_Z) / \operatorname{row}(H_X)$'''
    n = ker_of.shape[1]
    candidates = null(ker_of)
    R_mod, piv_mod = rref(mod_by)
    ech = [(row.copy(), p) for row, p in zip(R_mod, piv_mod)]
    kept = []
    for v in candidates:
        w = v.copy()
        for row, p in ech:
            if w[p]:
                w ^= row
        nz = np.flatnonzero(w)
        if len(nz):
            ech.append((w, int(nz[0])))
            kept.append(v)
    return np.array(kept, dtype=np.uint8) if kept else np.zeros((0, n), dtype=np.uint8)

def obs(H_X: np.ndarray, H_Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r'''
    Computes conjugate CSS logical operator bases $(L_X, L_Z)$ over $\mathbb{F}_2$:
    $$L_Z = \ker(H_X) / \operatorname{row}(H_Z), \quad L_X = \ker(H_Z) / \operatorname{row}(H_X)$$
    satisfying $L_X L_Z^T = I_k \pmod 2$.
    '''
    assert np.all((H_X @ H_Z.T) % 2 == 0), 'H_X and H_Z do not commute!'
    n = H_X.shape[1]

    L_Z = quot(H_X, H_Z)
    L_X_raw = quot(H_Z, H_X)
    assert L_X_raw.shape[0] == L_Z.shape[0], 'CSS logical dimension mismatch!'

    if L_Z.shape[0] == 0:
        return L_X_raw, L_Z

    M = (L_X_raw @ L_Z.T) % 2
    L_X = (inv(M) @ L_X_raw) % 2
    return L_X, L_Z
