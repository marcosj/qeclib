import numpy as np
#import stim
from circ import css2stim
from noise import add_noise
from tanner import export_fpga, quantize_llr

# [7, 4, 3] Hamming parity check matrix
'''H = np.array([
    [1,0,0,1,0,1,1],
    [0,1,0,1,1,0,1],
    [0,0,1,0,1,1,1]
], dtype=np.uint8)'''

H = np.array([
    [0,0,0,1,1,1,1],
    [0,1,1,0,0,1,1],
    [1,0,1,0,1,0,1]
], dtype=np.uint8)

c = css2stim(H_X=H, H_Z=H, rnd=1, mem='Z')
svg = c.diagram('timeslice-svg') # detslice-with-ops
with open('out/steane.svg', 'w') as f: f.write(str(svg))
dem = c.detector_error_model() #decompose_errors=True)

# [[7, 1, 3]] Steane CSS code
steane = css2stim(H_X=H, H_Z=H, rnd=3, mem='Z')

# Inject SI1000 noise model (p = 0.002)
p = 0.002
si1000 = add_noise(
    steane,
    p_1q=p / 10,
    p_2q=p,
    p_pp=2 * p,
    p_ro=5 * p,
    p_iq=p / 10,
    p_im=2 * p,
    bias=0.0,
)
si1000.to_file('out/steane.stim')

# Inject Quantum-Dot echoed noise (T_ro = 2us, T2_DD = 200us, 90% Z-biased)
'''p_idle_qd = 2e-6 / (2 * 200e-6)  # 0.005
qdot = add_noise(
    steane,
    p_1q=0.0005,
    p_2q=0.003,
    p_pp=0.004,
    p_ro=0.01,
    p_iq=0.0002,
    p_im=p_idle_qd,
    bias=0.90,
) not right now'''

tg = export_fpga(
    si1000,
    out_dir='out',
    shots=10_000,
    frac_bits=4,
    total_bits=8,
    seed=42,
)
llr_q = quantize_llr(tg.llr, frac_bits=4, total_bits=8)

print(f'Exported to out/: checks={tg.H.shape[0]}, vars={tg.H.shape[1]}, edges={int(tg.H.sum())}')
print(f'Check deg min/max: {tg.H.sum(axis=1).min()} / {tg.H.sum(axis=1).max()}')
print(f'Var deg min/max:   {tg.H.sum(axis=0).min()} / {tg.H.sum(axis=0).max()}')
print(f'LLR int8 min/max:  {llr_q.min()} / {llr_q.max()}')

# Sample bit-packed syndrome frames for FPGA testbench!
#sampler = si1000.compile_detector_sampler()
#syndromes = sampler.sample(shots=100_000, bit_packed=True)
#print(f'Generated FPGA test vectors: shape={syndromes.shape}, dtype={syndromes.dtype}')
#dets, obs_flips = sampler.sample(shots=10_000, separate_observables=True)
#print(f'Sampled dets: {dets.shape}, obs: {obs_flips.shape}')
