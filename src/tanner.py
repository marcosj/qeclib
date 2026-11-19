import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import stim
import networkx as nx
import matplotlib.pyplot as plt


@dataclass
class TannerGraph:
    r"""
    Container for a circuit-level DEM Tanner graph and FPGA decoder inputs.

    Attributes:
        H: Parity-check matrix of shape `(num_checks, num_vars)` ($\mathbb{F}_2$).
        L: Logical observable flip mask of shape `(num_obs, num_vars)` ($\mathbb{F}_2$).
        probs: Fault mechanism probabilities of shape `(num_vars,)`.
        llr: Floating-point channel LLRs $\ln\frac{1 - p}{p}$ of shape `(num_vars,)`.
        check_to_vars: Adjacency list mapping `check_id -> list[var_id]`.
        var_to_checks: Adjacency list mapping `var_id -> list[check_id]`.
        det_coords: Detector `(x, y, t, ...)` coordinates indexed by `check_id`.
    """
    H: np.ndarray
    L: np.ndarray
    probs: np.ndarray
    llr: np.ndarray
    check_to_vars: dict[int, list[int]]
    var_to_checks: dict[int, list[int]]
    det_coords: dict[int, list[float]]


def build_tanner_graph(H: np.ndarray) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """
    Builds integer adjacency lists from a binary parity-check matrix `H`.

    Args:
        H: Binary matrix of shape `(num_checks, num_vars)`.

    Returns:
        `(check_to_vars, var_to_checks)` covering every check and variable index.
    """
    num_checks, num_vars = H.shape
    check_to_vars = {c: [] for c in range(num_checks)}
    var_to_checks = {v: [] for v in range(num_vars)}

    rows, cols = np.where(H == 1)
    for r, c in zip(rows, cols):
        check_to_vars[int(r)].append(int(c))
        var_to_checks[int(c)].append(int(r))

    return check_to_vars, var_to_checks

def quantize_llr(
    llr: np.ndarray,
    frac_bits: int = 4,
    total_bits: int = 8,
    min_pos: int = 1,
) -> np.ndarray:
    r"""
    Quantizes floating-point channel LLRs to signed fixed-point integers for FPGA Min-Sum/BP.

    $$\text{LLR}_{\text{int}} = \operatorname{clip}\!\left(\operatorname{round}\!\left(\text{LLR} \cdot 2^{\text{frac\_bits}}\right),\, \text{min\_pos},\, 2^{\text{total\_bits}-1}-1\right)$$

    Args:
        llr: Floating-point LLR vector.
        frac_bits: Number of fractional bits (scale factor $2^{\text{frac\_bits}}$).
        total_bits: Signed integer bitwidth (e.g. `8` for `int8`, `16` for `int16`).
        min_pos: Minimum magnitude for positive LLRs so noisy edges don't round to `0`.

    Returns:
        Quantized integer NumPy array (`int8` if `total_bits <= 8` else `int16`/`int32`).
    """
    max_val = (1 << (total_bits - 1)) - 1
    scaled = np.rint(llr * (1 << frac_bits)).astype(np.int64)
    pos_mask = llr > 0
    scaled[pos_mask] = np.clip(scaled[pos_mask], min_pos, max_val)
    scaled[~pos_mask] = np.clip(scaled[~pos_mask], -(max_val + 1), max_val)

    dtype = np.int8 if total_bits <= 8 else (np.int16 if total_bits <= 16 else np.int32)
    return scaled.astype(dtype)

def dem_to_tanner(
    source: stim.Circuit | stim.DetectorErrorModel,
    decompose: bool = False,
) -> TannerGraph:
    r"""
    Converts a noisy Stim circuit or `DetectorErrorModel` into a `TannerGraph`.

    Each `error(p)` instruction in the flattened DEM becomes one variable node
    with channel LLR $\ln\frac{1 - p}{p}$, connected to its detector checks.

    Args:
        source: Either a noisy `stim.Circuit` or a `stim.DetectorErrorModel`.
        decompose: Passed to `circuit.detector_error_model(decompose_errors=...)`
            when `source` is a `stim.Circuit`. Keep `False` for hypergraph codes (Steane).

    Returns:
        Populated `TannerGraph` dataclass.
    """
    if isinstance(source, stim.Circuit):
        dem = source.detector_error_model(decompose_errors=decompose)
    else:
        dem = source

    flat = dem.flattened()
    num_checks = flat.num_detectors
    num_obs = flat.num_observables

    err_dets: list[list[int]] = []
    err_obs: list[list[int]] = []
    err_probs: list[float] = []

    for inst in flat:
        if inst.type != 'error':
            continue
        p = float(inst.args_copy()[0])
        targets = inst.targets_copy()
        dets = [t.val for t in targets if t.is_relative_detector_id()]
        obs_ids = [t.val for t in targets if t.is_logical_observable_id()]

        if not dets:
            continue  # undetectable fault (no check neighbors)

        err_dets.append(dets)
        err_obs.append(obs_ids)
        err_probs.append(p)

    num_vars = len(err_probs)
    H = np.zeros((num_checks, num_vars), dtype=np.uint8)
    L = np.zeros((num_obs, num_vars), dtype=np.uint8)

    for v, (dets, obs_ids) in enumerate(zip(err_dets, err_obs)):
        H[dets, v] = 1
        if obs_ids:
            L[obs_ids, v] = 1

    probs = np.clip(np.asarray(err_probs, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    llr = np.log((1.0 - probs) / probs)
    check_to_vars, var_to_checks = build_tanner_graph(H)
    det_coords = {int(k): list(v) for k, v in flat.get_detector_coordinates().items()}

    return TannerGraph(
        H=H,
        L=L,
        probs=probs,
        llr=llr,
        check_to_vars=check_to_vars,
        var_to_checks=var_to_checks,
        det_coords=det_coords,
    )

def build_dem_graph(
    H: np.ndarray,
    error_probs: np.ndarray,
    strict: bool = False,
) -> tuple[dict[int, list[tuple[int, float, int]]], int, int]:
    """
    Converts a graphlike parity-check matrix `H` (column weight <= 2) into a
    matching detector graph with virtual boundary node `num_checks`.

    Args:
        H: Binary parity-check matrix of shape `(num_checks, num_vars)`.
        error_probs: Fault probability for each variable column.
        strict: If `True`, raises `ValueError` when any variable touches > 2 checks.

    Returns:
        `(adj, boundary, skipped_hyperedges)` where `adj[u]` is a list of
        `(neighbor, weight, var_idx)` tuples.
    """
    num_checks, num_vars = H.shape
    boundary = num_checks
    adj: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
    skipped = 0

    probs = np.clip(error_probs, 1e-15, 0.5 - 1e-15)
    weights = np.log((1.0 - probs) / probs)

    for v in range(num_vars):
        checks = np.flatnonzero(H[:, v])
        w = float(weights[v])

        if len(checks) == 2:
            u, z = int(checks[0]), int(checks[1])
            adj[u].append((z, w, v))
            adj[z].append((u, w, v))
        elif len(checks) == 1:
            u = int(checks[0])
            adj[u].append((boundary, w, v))
            adj[boundary].append((u, w, v))
        elif len(checks) > 2:
            if strict:
                raise ValueError(f'Variable {v} has degree {len(checks)} > 2 (hyperedge).')
            skipped += 1

    return dict(adj), boundary, skipped

def plot_tanner_and_dem(H: np.ndarray, probs: np.ndarray, out_path: str = 'g.pdf') -> None:
    """Plots bipartite Tanner graph (left) and matching DEM graph if degree <= 2 (right)."""
    adj, boundary, skipped = build_dem_graph(H, probs, strict=False)
    probs_c = np.clip(probs, 1e-15, 1.0 - 1e-15)
    weights = np.log((1.0 - probs_c) / probs_c)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Plot 1: Bipartite Tanner Graph
    B = nx.Graph()
    checks = [f'C{i}' for i in range(H.shape[0])]
    vars_ = [f'V{j}' for j in range(H.shape[1])]
    B.add_nodes_from(checks, bipartite=0)
    B.add_nodes_from(vars_, bipartite=1)

    pos_tanner = {c: (i + 0.5, 1.0) for i, c in enumerate(checks)}
    pos_tanner.update({v: (j, 0.0) for j, v in enumerate(vars_)})

    for r, c in zip(*np.where(H == 1)):
        B.add_edge(f'C{r}', f'V{c}')

    nx.draw(
        B, pos_tanner, ax=ax1, with_labels=True,
        node_color=['#ff9999' if n in checks else '#99ccff' for n in B.nodes()],
        node_shape='s', node_size=900, font_weight='bold', edge_color='#555555', width=1.5,
    )
    ax1.set_title('Tanner Graph (Checks vs. Fault Variables)', fontsize=12, fontweight='bold')
    ax1.set_ylim(-0.4, 1.4)

    # Plot 2: DEM Matching Graph (if graphlike)
    if skipped == 0 and adj:
        G_dem = nx.Graph()
        b_label = f'B{boundary}'
        for v in range(H.shape[1]):
            active = np.flatnonzero(H[:, v])
            lbl = f'V{v}\nw={weights[v]:.2f}'
            if len(active) == 2:
                G_dem.add_edge(f'D{active[0]}', f'D{active[1]}', label=lbl)
            elif len(active) == 1:
                G_dem.add_edge(f'D{active[0]}', b_label, label=lbl)

        pos_dem = nx.circular_layout(G_dem)
        node_colors = ['#ffcc66' if n == b_label else '#ff9999' for n in G_dem.nodes()]
        nx.draw(
            G_dem, pos_dem, ax=ax2, with_labels=True,
            node_color=node_colors, node_size=1200, font_weight='bold', edge_color='#333333', width=1.5,
        )
        nx.draw_networkx_edge_labels(G_dem, pos_dem, edge_labels=nx.get_edge_attributes(G_dem, 'label'), ax=ax2, font_size=8)
        ax2.set_title('DEM Matching Graph (Detectors + Boundary)', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, f'Hypergraph DEM ({skipped} edges with deg > 2)\nUse Tanner graph!', ha='center', va='center')
        ax2.set_axis_off()

    plt.tight_layout()
    plt.savefig(out_path)

def export_fpga(
    circ: stim.Circuit,
    out_dir: str | Path = 'out',
    shots: int = 10_000,
    frac_bits: int = 4,
    total_bits: int = 8,
    decompose: bool = False,
    seed: int | None = 42,
) -> TannerGraph:
    """
    Exports Tanner adjacency lists, quantized LLRs, DEM, logical mask `L`,
    and bit-packed/unpacked syndrome test vectors for FPGA/RFSoC generation.

    Writes `model.json`, `vectors.npz`, and `model.dem` into `out_dir`.

    Args:
        circ: Noisy Stim circuit.
        out_dir: Output directory path.
        shots: Number of syndrome + observable test shots to sample.
        frac_bits: Fractional bits for fixed-point LLR quantization.
        total_bits: Signed integer bitwidth for quantized LLRs.
        decompose: Whether to decompose hyperedges in the DEM (`False` for Steane).
        seed: RNG seed for reproducible test vectors (`None` for random).

    Returns:
        The extracted `TannerGraph`.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dem = circ.detector_error_model(decompose_errors=decompose)
    tg = dem_to_tanner(dem)
    llr_int = quantize_llr(tg.llr, frac_bits=frac_bits, total_bits=total_bits)

    num_checks, num_vars = tg.H.shape
    num_obs = tg.L.shape[0]

    # Sample testbench vectors (single draw so packed & unpacked match 1-to-1)
    sampler = circ.compile_detector_sampler(seed=seed)
    dets, obs_flips = sampler.sample(shots=shots, separate_observables=True)
    dets = dets.astype(np.uint8)
    obs_flips = obs_flips.astype(np.uint8)
    dets_packed = np.packbits(dets, axis=1, bitorder='little')
    obs_packed = np.packbits(obs_flips, axis=1, bitorder='little')

    # CSR representation (handy for BRAM / HLS / Verilog generation)
    check_ptr = np.zeros(num_checks + 1, dtype=np.int32)
    check_ptr[1:] = np.cumsum([len(tg.check_to_vars[c]) for c in range(num_checks)])
    check_idx = np.array(
        [v for c in range(num_checks) for v in tg.check_to_vars[c]], dtype=np.int32
    )

    var_ptr = np.zeros(num_vars + 1, dtype=np.int32)
    var_ptr[1:] = np.cumsum([len(tg.var_to_checks[v]) for v in range(num_vars)])
    var_idx = np.array(
        [c for v in range(num_vars) for c in tg.var_to_checks[v]], dtype=np.int32
    )

    # 1. Raw Stim DEM
    (out / 'model.dem').write_text(str(dem))

    # 2. JSON topology & fixed-point config (zero NumPy needed on RFSoC side)
    model_json = {
        'num_checks': int(num_checks),
        'num_vars': int(num_vars),
        'num_obs': int(num_obs),
        'num_edges': int(tg.H.sum()),
        'max_check_deg': int(tg.H.sum(axis=1).max()) if num_checks else 0,
        'max_var_deg': int(tg.H.sum(axis=0).max()) if num_vars else 0,
        'llr_frac_bits': int(frac_bits),
        'llr_total_bits': int(total_bits),
        'check_to_vars': [tg.check_to_vars[c] for c in range(num_checks)],
        'var_to_checks': [tg.var_to_checks[v] for v in range(num_vars)],
        'obs_mask': tg.L.tolist(),
        'obs_vars': [np.flatnonzero(tg.L[k]).tolist() for k in range(num_obs)],
        'llr_int': llr_int.tolist(),
        'llr_float': tg.llr.tolist(),
        'probs': tg.probs.tolist(),
    }
    (out / 'model.json').write_text(json.dumps(model_json, indent=2))

    # 3. Binary NumPy bundle (CSR + dense + 10k test vectors)
    np.savez_compressed(
        out / 'vectors.npz',
        H=tg.H,
        L=tg.L,
        probs=tg.probs,
        llr=tg.llr,
        llr_int=llr_int,
        check_ptr=check_ptr,
        check_idx=check_idx,
        var_ptr=var_ptr,
        var_idx=var_idx,
        dets=dets,
        obs_flips=obs_flips,
        dets_packed=dets_packed,
        obs_packed=obs_packed,
    )

    return tg
