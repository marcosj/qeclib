# qeclib

A lightweight Python toolkit for building CSS quantum error-correcting circuits in [Stim](https://github.com/quantumlib/Stim), injecting circuit-level noise, and exporting Detector Error Model (DEM) hypergraph Tanner structures and fixed-point test vectors for hardware (FPGA/RFSoC) and software decoders.

## Features

- **$\mathbb{F}_2$ CSS Algebra (`src/gf2.py`):** Gauss-Jordan elimination (`rref`, `null`, `inv`) and automatic conjugate logical operator synthesis ($L_X L_Z^T = I_k \pmod 2$) via `obs(H_X, H_Z)`.
- **Optimal CNOT Scheduling & Stim Circuits (`src/circ.py`):**
  - `h2cnots`: König / Kempe-chain bipartite edge coloring that guarantees minimum-depth $\Delta(H)$ parallel CNOT layers.
  - `css2stim`: Generates $X/Z$-memory Stim circuits with `QUBIT_COORDS`, space-time `DETECTOR` + `SHIFT_COORDS`, and `OBSERVABLE_INCLUDE`.
- **Circuit-Level Noise (`src/noise.py`):** Tick-by-tick `add_noise` supporting SI1000 and configurable 1q/2q gate, reset, measurement, and idle error channels.
- **DEM $\to$ Tanner Graph & FPGA Export (`src/tanner.py`):**
  - `dem_to_tanner`: Parses hypergraph DEMs (degree $\ge 1$) into `H`, logical flip mask `L`, probabilities, floating-point LLRs, and `check_to_vars` / `var_to_checks` adjacency lists.
  - `quantize_llr`: Signed fixed-point LLR quantization (`int8` / `int16`) for Min-Sum / BP hardware.
  - `export_fpga`: Exports topology, quantized parameters, CSR indices, and bit-packed/unpacked syndrome shots.
  - `build_dem_graph` / `plot_tanner_and_dem`: Bipartite Tanner and $\le 2$-edge matching-graph visualization.

## Repo Layout

```text
qeclib/
├── Makefile
├── requirements.txt
├── docs/               # Auto-generated API docs (pdoc + MathJax)
└── src/
    ├── gf2.py          # GF(2) linear algebra & CSS logicals
    ├── circ.py         # Optimal CNOT scheduler & Stim CSS circuit builder
    ├── noise.py        # Tick-level circuit noise injection
    ├── tanner.py       # DEM -> Tanner graph, LLR quantization & FPGA export
    └── main.py         # Steane [[7, 1, 3]] + SI1000 export example
```

## Quickstart

```bash
pip install -r requirements.txt
make run
```

Running `make run` builds a 3-round `[[7, 1, 3]]` Steane memory-$Z$ circuit under SI1000 noise ($p = 0.002$) and exports to `out/`:

| File | Contents |
| :--- | :--- |
| `out/model.json` | `check_to_vars`, `var_to_checks`, `llr_int`, `obs_mask`, `obs_vars`, and graph sizing metadata (`num_checks=18`, `num_vars=276`, `num_edges=881`, `max_check_deg=79`, `max_var_deg=6`). |
| `out/vectors.npz` | Dense `H`, `L`, `probs`, `llr`, `llr_int`, CSR arrays (`check_ptr`/`check_idx`, `var_ptr`/`var_idx`), and `10,000` sampled shots (`dets`, `obs_flips`, little-endian `dets_packed`, `obs_packed`). |
| `out/model.dem` | Raw flattened Stim `DetectorErrorModel`. |

## API Documentation

Generate local HTML API docs into `docs/`:

```bash
make docs
```

## Roadmap

- Reference software decoders (Min-Sum, Normalized/Offset Min-Sum, BP-OSD, MWPM)
- Exact biased Pauli noise channels (`PAULI_CHANNEL_1`) for spin-qubit / quantum-dot hardware
- Automated test suite across surface, color, and CSS QLDPC codes
