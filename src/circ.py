import numpy as np
import stim
from gf2 import obs

def h2cnots(H: np.ndarray, a0: int, isZ: bool) -> list[list[tuple[int, int]]]:
    """
    Optimal bipartite edge-coloring CNOT scheduler (Kempe-chain augmenting paths).
    Guarantees exactly max_degree(H) parallel, non-overlapping CNOT layers.
    Returns a list of layers, where each layer is a list of (control, target) pairs.
    """
    if not np.any(H):
        return []

    aq, dq = H.shape
    num_colors = int(max(H.sum(axis=1).max(), H.sum(axis=0).max()))

    # adj[node][color] = neighbor, keeping ancilla ('a', a) and data ('d', d) disjoint
    adj: dict[tuple[str, int], dict[int, tuple[str, int]]] = {
        **{('a', a): {} for a in range(aq)},
        **{('d', d): {} for d in range(dq)},
    }

    for a in range(aq):
        for d in np.flatnonzero(H[a]):
            u, v = ('a', a), ('d', int(d))
            free_u = [c for c in range(num_colors) if c not in adj[u]]
            free_v = [c for c in range(num_colors) if c not in adj[v]]
            common = [c for c in free_u if c in free_v]

            if common:
                c = common[0]
                adj[u][c] = v
                adj[v][c] = u
            else:
                # Flip an alternating (c1, c2) Kempe path starting at v;
                # bipartiteness + c1 free at u guarantees the path never reaches u.
                c1, c2 = free_u[0], free_v[0]
                path = []
                curr, want = v, c1
                while want in adj[curr]:
                    nxt = adj[curr][want]
                    path.append((curr, nxt, want))
                    curr = nxt
                    want = c2 if want == c1 else c1

                for x, y, col in path:
                    del adj[x][col]
                    del adj[y][col]
                for x, y, col in path:
                    new_col = c2 if col == c1 else c1
                    adj[x][new_col] = y
                    adj[y][new_col] = x

                adj[u][c1] = v
                adj[v][c1] = u

    layers: list[list[tuple[int, int]]] = [[] for _ in range(num_colors)]
    for a in range(aq):
        ai = a0 + a
        for c, (_, d) in adj[('a', a)].items():
            cnot = (d, ai) if isZ else (ai, d)
            layers[c].append(cnot)

    return [sorted(layer) for layer in layers if layer]

def css2stim(H_X: np.ndarray, H_Z: np.ndarray, rnd: int = 3, mem: str = 'Z', L = None) -> stim.Circuit:
    '''
    Creates Stim memory `mem` circuit from CSS parity checks (H_X, H_Z).
    Qubit layout:
      [0 .. n-1]               -> Data qubits
      [n .. n+mx-1]            -> X-check ancillas
      [n+mx .. n+mx+mz-1]      -> Z-check ancillas
    '''
    assert np.all((H_X @ H_Z.T) % 2 == 0), 'H_X and H_Z do not commute!'
    assert mem in ('Z', 'X'), "mem must be 'Z' or 'X'"
    assert rnd >= 1, 'rnd must be >= 1'

    if L is None:
        L_X, L_Z = obs(H_X, H_Z)
        L_mem = L_Z if mem == 'Z' else L_X
    else:
        L_mem = np.atleast_2d(L).astype(np.uint8)

    n  = H_X.shape[1]
    mx = H_X.shape[0]
    mz = H_Z.shape[0]

    dq = list(range(n))
    xa = list(range(n, n + mx))
    za = list(range(n + mx, n + mx + mz))
    aq = xa + za

    x_layers = h2cnots(H_X, a0=n, isZ=False)
    z_layers = h2cnots(H_Z, a0=n + mx, isZ=True)

    c = stim.Circuit()

    dq_coords = [(float(i), 1.0) for i in range(n)]
    xa_coords = [(i + (n - mx) / 2.0, 0.0) for i in range(mx)]
    za_coords = [(i + (n - mz) / 2.0, 2.0) for i in range(mz)]

    for q, xy in zip(dq, dq_coords):
        c.append('QUBIT_COORDS', q, xy)
    for q, xy in zip(xa, xa_coords):
        c.append('QUBIT_COORDS', q, xy)
    for q, xy in zip(za, za_coords):
        c.append('QUBIT_COORDS', q, xy)

    # Preparation
    if mem == 'Z':
        c.append('R', dq + aq)
    else:
        c.append('RX', dq)
        c.append('R', aq)

    # Syndrome extraction rounds
    for r in range(rnd):
        c.append('TICK')

        # Prepare X-ancillas in |+>
        c.append('H', xa)
        c.append('TICK')

        # Execute scheduled X-check and Z-check CNOT layers
        for layer in x_layers:
            for u, v in layer:
                c.append('CX', [u, v])
            c.append('TICK')

        for layer in z_layers:
            for u, v in layer:
                c.append('CX', [u, v])
            c.append('TICK')

        # Rotate X-ancillas back to Z-basis
        c.append('H', xa)
        c.append('TICK')

        # Measure and reset all ancillas
        c.append('MR', aq)

        # Define space-time detectors (XOR current ancilla with previous round)
        na = mx + mz
        for i in range(na):
            rec = -na + i
            x, y = xa_coords[i] if i < mx else za_coords[i - mx]
            if r == 0:
                if (mem == 'Z' and i >= mx) or (mem == 'X' and i < mx):
                    c.append('DETECTOR', [stim.target_rec(rec)], [x, y, 0.0])
            else:
                prev = -2 * na + i
                c.append('DETECTOR', [stim.target_rec(rec), stim.target_rec(prev)], [x, y, 0.0])

        c.append('SHIFT_COORDS', [], [0.0, 0.0, 1.0])

    # Final readout (shares the final TICK with round rnd-1's MR)
    if mem == 'Z':
        c.append('M', dq)
        for a in range(mz):
            tgt = [stim.target_rec(-n + d) for d in np.where(H_Z[a] == 1)[0]]
            tgt.append(stim.target_rec(-n - mz + a))
            x, y = za_coords[a]
            c.append('DETECTOR', tgt, [x, y, 0.0])
    else:
        c.append('MX', dq)
        for a in range(mx):
            tgt = [stim.target_rec(-n + d) for d in np.where(H_X[a] == 1)[0]]
            tgt.append(stim.target_rec(-n - (mx + mz) + a))
            x, y = xa_coords[a]
            c.append('DETECTOR', tgt, [x, y, 0.0])

    for obs_idx, row in enumerate(L_mem):
        tgt = [stim.target_rec(-n + d) for d in np.where(row == 1)[0]]
        c.append('OBSERVABLE_INCLUDE', tgt, obs_idx)

    return c
