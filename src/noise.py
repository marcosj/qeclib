import stim

def add_noise(
    circ: stim.Circuit,
    p_1q: float,
    p_2q: float,
    p_pp: float,
    p_ro: float,
    p_iq: float,
    p_im: float,
    bias: float = 0.0,  # 0.0 = symmetric depolarizing; >0 = extra Z-dephasing (quantum dots)
) -> stim.Circuit:
    '''
    Adds custom circuit level noise or SI1000.
    Walks a tick-delimited noiseless Stim c and injects gate, measurement,
    reset, and exact per-TICK idle noise channels.
    '''
    all_qubits = set(range(circ.num_qubits))
    noisy = stim.Circuit()

    # Group instructions by TICK
    ticks = []
    current_tick = []
    for inst in circ.flattened():
        if inst.name == 'TICK':
            ticks.append(current_tick)
            current_tick = []
        else:
            current_tick.append(inst)
    if current_tick:
        ticks.append(current_tick)

    METADATA = ('QUBIT_COORDS', 'SHIFT_COORDS', 'DETECTOR', 'OBSERVABLE_INCLUDE')
    for tick_insts in ticks:
        active_qubits = set()
        has_meas_or_reset = False

        # First pass: identify active qubits and operation types in this TICK
        for inst in tick_insts:
            if inst.name in METADATA:
                continue
            tgt = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            active_qubits.update(tgt)
            if inst.name in ('M', 'MX', 'R', 'RX', 'MR'):
                has_meas_or_reset = True

        # Second pass: emit instructions + active gate/meas/reset noise
        for inst in tick_insts:
            if inst.name in METADATA:
                noisy.append(inst)
                continue

            tgt = [t.value for t in inst.targets_copy() if t.is_qubit_target]

            # Pre-measurement bit-flip error
            if inst.name in ('M', 'MR') and p_ro > 0:
                noisy.append('X_ERROR', tgt, p_ro)
            elif inst.name == 'MX' and p_ro > 0:
                noisy.append('Z_ERROR', tgt, p_ro)

            # Emit the ideal gate
            noisy.append(inst)

            # Post-gate / post-reset noise
            if inst.name in ('R', 'MR') and p_pp > 0:
                noisy.append('X_ERROR', tgt, p_pp)
            elif inst.name == 'RX' and p_pp > 0:
                noisy.append('Z_ERROR', tgt, p_pp)
            elif inst.name == 'H' and p_1q > 0:
                noisy.append('DEPOLARIZE1', tgt, p_1q)
            elif inst.name == 'CX' and p_2q > 0:
                noisy.append('DEPOLARIZE2', tgt, p_2q)

        # Third pass: inject idle noise on all untouched qubits in this TICK
        idle_qubits = sorted(all_qubits - active_qubits)
        if idle_qubits:
            p_idle = p_im if has_meas_or_reset else p_iq
            if p_idle > 0:
                if bias > 0:
                    # Biased quantum-dot model: split into Z-dephasing + residual depolarizing
                    p_z = p_idle * bias
                    p_dep = p_idle * (1.0 - bias)
                    if p_z > 0:
                        noisy.append('Z_ERROR', idle_qubits, p_z)
                    if p_dep > 0:
                        noisy.append('DEPOLARIZE1', idle_qubits, p_dep)
                else:
                    noisy.append('DEPOLARIZE1', idle_qubits, p_idle)

    return noisy
