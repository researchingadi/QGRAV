# QGRAV: Complete Research Documentation

> *This file is the living record of every decision made in the QGRAV project — the reasoning, the alternatives considered, the literature that shaped our thinking, and the precise evolution of our contribution statement. It is updated continuously as the project progresses. Anyone reading this document should be able to reconstruct every significant choice we made and understand exactly why we made it.*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Genesis: How QGRAV Was Born](#2-genesis-how-qgrav-was-born)
3. [Chronological Decision Timeline](#3-chronological-decision-timeline)
4. [Architecture Decision Record](#4-architecture-decision-record)
5. [Preprocessing Pipeline Contract](#5-preprocessing-pipeline-contract)
6. [Literature Review Record](#6-literature-review-record)
7. [Novel Contribution Statement (Current)](#7-novel-contribution-statement-current)
8. [Publication Strategy](#8-publication-strategy)
9. [Platform and Infrastructure Decisions](#9-platform-and-infrastructure-decisions)
10. [Open Questions and Future Work](#10-open-questions-and-future-work)

---

## 1. Project Overview

**Full title:** QGRAV: Hybrid Quantum-Classical Graph Attention Network for Gravitational Wave Detection

**Author:** Adi Singh, Mississippi State University (B.S. Computer Science; M.S. Computer Science, Fall 2026)

**Target venue:** *Machine Learning: Science and Technology* (IOP Publishing) — indexed in Web of Science and Scopus, open access, specifically created for ML methods applied to physical science problems.

**Repository:** `github.com/researchingadi/QGRAV`

**Current status:** Pre-code. Architecture fully designed, preprocessing contract locked, literature review complete, documentation in progress. Data pipeline is next.

**The one-sentence description:**
QGRAV is the first architecture to encode the measured Pearson cross-correlation coherence and inter-detector time-delay geometry as physics-informed graph edge features in a GATv2 attention layer, enabling event-specific detector weighting as a learned function of measured network coherence rather than a fixed aggregation rule.

---

## 2. Genesis: How QGRAV Was Born

### May 7, 2026 — Initial Conception

The project began with a goal: build a project involving gravitational wave detection in noisy data. The initial observation that formed the project's core contribution was this:

> *Every deep learning paper on gravitational wave detection since George & Huerta (2018) processes each LIGO/Virgo detector independently and fuses results with naive concatenation or majority voting. This discards the single most physically meaningful constraint in multi-detector GW astronomy: the inter-detector propagation delay is a hard geometric prior that encodes sky position, and the cross-detector coherence of a signal is literally how physicists confirm a real detection.*

This observation directly motivated the graph neural network framing: model detectors as nodes, encode physical constraints as edge features, and let a graph attention network learn which inter-detector relationships matter for each candidate event.

### The Three-Layer Architecture Insight

The project was designed from the start around three stacked contributions, each building on the last:

1. **Multi-detector graph structure** — detectors as nodes, physical time-delay constraints as edges (the structural innovation)
2. **Physics-informed edge features** — cross-correlation lag and coherence score computed from measured data (the physical prior)
3. **Variational quantum circuit classifier** — benchmarking quantum feature spaces on real GW data (the exploratory extension)

The quantum component was deliberately framed as exploratory from day one, with explicit acknowledgment that NISQ hardware is unlikely to outperform a classical MLP head at this scale. The publishable contribution of Phase 3 is the *benchmark itself* — hardware noise characterization on real LIGO data — regardless of whether the VQC wins or loses.

### The Initial Title

Early working title: *GravDetect — Multi-Detector Graph Attention Network for Gravitational Wave Detection*

Later renamed QGRAV after the quantum component was added.

---

## 3. Chronological Decision Timeline

### May 7, 2026 — Core Architecture Established

**Decision:** Model the LIGO-Virgo detector network as a graph with physical time-delay edge features and train a Graph Attention Network.

**Rationale:** The inter-detector propagation delay (H1→L1 ≈ 10ms, H1→V1 ≈ 27ms) encodes sky position geometry. A real gravitational wave must arrive at each detector with a time delay consistent with wave propagation at the speed of light. A glitch appearing in only one detector has no coherent counterpart. Encoding this constraint as a graph prior gives the model physics knowledge that no prior deep learning GW paper had built in architecturally.

**What we explicitly ruled out at this stage:** Simple concatenation of detector outputs (the standard approach), fixed-weight averaging, majority voting, and per-detector independent classification.

---

### May 7, 2026 — Quantum Component Added

**Decision:** Replace the final classification head with a Variational Quantum Circuit (VQC) using Qiskit's ZZFeatureMap + RealAmplitudes ansatz, connected to PyTorch via TorchConnector, trained on AerSimulator and validated on IBM Quantum hardware.

**Rationale:** This creates a genuinely novel hybrid architecture — quantum classifier receiving graph-contextualized multi-detector features — that has not appeared in the GW detection literature. The contribution is framed not as quantum advantage (which NISQ hardware cannot deliver at this scale) but as the first rigorous VQC benchmark on real LIGO data with hardware noise characterization.

**Key implementation detail:** ZZFeatureMap (8 qubits, 2 reps) + RealAmplitudes (8 qubits, 3 reps). Linear bridge layer W_bridge ∈ ℝ^{8×256} compresses GAT output to 8 dimensions for qubit encoding. Parameter-shift rule for gradients.

---

### May 7, 2026 — Platform Selection

**Decision:** Lightning AI over Google Colab.

**Why not Colab:** LIGO data must be re-downloaded every session (~20 minutes of setup). PyCBC conda installation rebuilds from scratch every session. Persistent `/teamspace` storage on Lightning AI eliminates both problems. Data downloaded once, environment installed once.

**Practical importance:** PyCBC requires conda (LALSuite has C/Fortran dependencies pip cannot resolve). This is non-negotiable.

---

### May 8, 2026 — Matched Filtering Framing Established

**Decision:** Frame matched filtering as a calibration anchor, not a competitor.

**The reasoning:** Matched filtering is the Neyman-Pearson optimal detector under Gaussian noise assumptions. A neural network cannot beat the theoretically optimal detector on the problem it was designed for. Claiming otherwise invites immediate reviewer rejection.

**The correct multi-objective framing:**
- **Sensitivity equality:** QGRAV achieves matched filtering sensitivity on clean signals
- **Glitch robustness:** QGRAV outperforms matched filtering on non-Gaussian noise (glitches), because it learned the glitch distribution implicitly from real noise
- **Sub-millisecond latency:** QGRAV enables real-time triggering for multimessenger follow-up networks (GOTO, BlackGEM) in the BNS merger scenario, where the kilonova blue component peaks within hours of merger

**The evaluation table that this produces:**

| Method | Sensitivity (Clean) | FAR (Glitchy Noise) | Latency |
|---|---|---|---|
| Matched filtering | Optimal (baseline) | High (glitch-prone) | Seconds–minutes |
| Single-detector CNN | High | Moderate | Sub-ms |
| QGRAV (ours) | Matches MF | Low (graph coherence rejects glitches) | Sub-ms |

---

### May 8, 2026 — Cross-Correlation Edge Feature Design

**Decision:** Use the measured inter-detector cross-correlation lag and peak coherence as the 2-element edge feature vector: e_{ij} = [τ_peak, |R_{ij}(τ_peak)|].

**What we rejected:** A static maximum time delay (e.g., 10ms for H1–L1). This would be physically wrong because the actual time delay depends on sky position and varies per event. A static scalar forces the model to assume all events come from the same sky direction — a non-physical prior.

**Why the cross-correlation:** The Pearson cross-correlation of whitened strain between two detectors gives:

1. **τ_peak** — the lag at maximum correlation, which encodes the actual measured time delay consistent with the event's sky geometry
2. **|R(τ_peak)|** — the peak correlation value, which encodes network coherence strength: high for real GW signals (same chirp in both detectors), near-zero for single-detector glitches

**The critical constraint:** Cross-correlation must be computed AFTER the 0.5-second edge crop of the whitened buffer — not before. Symmetric Tukey taper artifacts on the edges of uncropped windows create artificial correlation at Δt=0, which would blind the network to the true astrophysical time delay. This was identified as a trap that ruins approximately half of all early-stage physics-informed ML projects.

**Physical search windows:** τ_peak is restricted to ±10ms for H1–L1 and ±27ms for H1–V1 and L1–V1, corresponding to the maximum light-travel time between detector pairs.

**PyTorch Geometric implementation:** GATv2Conv with `edge_dim=2`. Confirmed: PyTorch Geometric's GATv2Conv explicitly supports multidimensional edge features through the `edge_dim` parameter, and the attention coefficient computation includes transformed edge attributes. Edge features enter the unnormalized attention score as: `e_{ij} = a^T * LeakyReLU(W * [h_i || h_j || W_e * edge_feat_ij])`.

---

### May 12, 2026 — Project Documentation System Established

**Decision:** Maintain a CLAUDE.md for session continuity plus a full DOCUMENTATION.md for research record. This file.

---

### June 2026 — Preprocessing Contract Locked In

*(See Section 5 for full details.)*

**High-level decision:** Hybrid preprocessing architecture. gwpy handles GWOSC data fetching and GPS time management only. All DSP (bandpass filtering, PSD estimation, tapering, whitening, cropping, cross-correlation) is implemented explicitly in scipy and NumPy.

**Why not gwpy for DSP:** gwpy's `.whiten()` and `.bandpass()` are black boxes. The exact tapering, PSD estimation choices, and crop behavior cannot be explained precisely to reviewers. For a paper targeting ML:ST, every parameter choice must be citable and defensible. scipy implementations are explicit and reproducible.

---

### June 2026 — Architecture Simplification Decision

**Decision:** Start Phase 1 with 1D CNN encoder (not 2D CNN/Q-transform), and H1+L1 only (not H1+L1+V1).

**Rationale for 1D CNN:**
- Simpler, faster to debug
- Most published papers (AResGW, Aframe) use 1D whitened strain directly
- 2D CNN on Q-transform spectrograms can be introduced as Phase 2 ablation: "does the richer time-frequency representation help?"

**Rationale for H1+L1 first:**
- Virgo has measurably lower sensitivity during O3 (BNS inspiral range: H1 135 Mpc, L1 115 Mpc, V1 50 Mpc)
- Starting with the two strongest detectors establishes the core claim cleanly
- Adding V1 in Phase 2 and testing whether the GAT learns to downweight it (appropriately, given its lower sensitivity) becomes a separate scientific result

---

### June 2026 — Injection Placement Decision

**Decision:** Randomize merger epoch uniformly within the central 0.5 seconds of the 1-second analysis window (merger time ∈ [0.25s, 0.75s]).

**Why not center-fixed injection:**
1. **Translation invariance:** Fixed placement at sample 1024 trains the CNN to fire on position, not morphology
2. **Physical time delays:** With fixed placement, the relative arrival time between H1 and L1 is always zero — no physical inter-detector delay variation. The GAT's cross-correlation edge feature would carry no geometric information
3. **Sliding-window deployment:** In real inference, the GW event falls at an unknown position within any given window. The model must generalize across positions

**Why the 0.5s central restriction:** Ensures inspiral tails and post-merger ringdown never bleed into the 0.5-second edge-crop zones, preventing signal truncation by the whitening filter.

---

### June 2026 — GATv2 vs Original GAT Selection

**Decision:** Use GATv2Conv (Brody et al. 2022), not the original GATConv (Veličković et al. 2018).

**The technical difference:**

Original GAT attention:
```
e_{ij} = a^T [Wh_i || Wh_j]
```
The same projection W is applied to both source and target before the attention vector, making attention "static" — for certain graph topologies, the same node pair always gets the same attention weight regardless of input context.

GATv2 attention:
```
e_{ij} = a^T LeakyReLU(W[h_i || h_j || W_e * edge_feat])
```
The concatenation and linear projection happen together, before the non-linearity, making attention "dynamic" — the same detector pair gets different weights depending on the current event's coherence characteristics. This is the physically correct behavior: H1-L1 attention should be high for a loud coherent BBH but lower if the detectors are currently in different noise states.

**PyTorch Geometric:** `GATv2Conv(in_channels, out_channels, heads=4, edge_dim=2)`.

---

### June 2026 — Literature Review Completed

*(See Section 6 for full details.)*

The literature review significantly refined the novelty claim. Before the review, we would have said "first GNN for multi-detector GW detection." After the review, the correct claim is more precise and more defensible.

---

## 4. Architecture Decision Record

### The Full Model Architecture (Phase 1)

```
Input: H1 whitened strain (2048 samples, 1s @ 2048Hz)
       L1 whitened strain (2048 samples, 1s @ 2048Hz)
       Edge feature e_H1L1 = [τ_peak, |R(τ_peak)|] ∈ ℝ^2

Per-detector encoding:
  CNN encoder (shared weights) → h_H1 ∈ ℝ^256
  CNN encoder (shared weights) → h_L1 ∈ ℝ^256

Graph construction:
  Nodes: {H1, L1} with features {h_H1, h_L1}
  Edges: {(H1→L1), (L1→H1), (H1→H1 self-loop), (L1→L1 self-loop)}
  Edge attributes: e_ij = [τ_peak, |R(τ_peak)|] for inter-detector edges; zeros for self-loops

GATv2 layer:
  4 attention heads, edge_dim=2
  Output: h'_H1 ∈ ℝ^256, h'_L1 ∈ ℝ^256

Graph readout:
  Mean pool across nodes → h_G ∈ ℝ^256

Classification:
  Phase 1/2: nn.Linear(256, 1) → sigmoid → p(GW) ∈ [0,1]
  Phase 3:   nn.Linear(256, 8) → ZZFeatureMap(8q) + RealAmplitudes → sigmoid → p(GW)
```

### CNN Encoder Architecture

Three convolutional blocks (Conv1D → BatchNorm → ReLU → MaxPool), followed by a linear projection to 256 dimensions.

**Why shared weights:** The same gravitational wave chirp morphology should produce the same feature vector regardless of which detector observed it. Shared weights enforce this physical symmetry and reduce parameter count.

**Input dimension:** 2048 samples (1 second at 2048 Hz after preprocessing).

### Self-Loop Necessity

Self-loops are mandatory in the 2-node Phase 1 graph. Without self-loops, each node has exactly one neighbor, and softmax over a single element always returns 1.0. There is no competition between "attend to self" and "attend to neighbor," so the attention mechanism is degenerate — it provides no learned weighting.

With self-loops, the softmax weighs two options:
- α_{H1,H1}: how much should H1's representation incorporate H1's own features (self-attention)
- α_{H1,L1}: how much should H1's representation incorporate L1's features (cross-detector attention)

For a real GW: high |R| → high α_{H1,L1} → cross-detector aggregation → representation captures network coherence
For a glitch in H1 only: low |R| → high α_{H1,H1} → H1 ignores L1 → glitch stays local

**This is the glitch rejection mechanism.** It is not hardcoded — it emerges from the physics-informed edge features.

---

## 5. Preprocessing Pipeline Contract

*This section records the exact preprocessing parameters, each cited to the literature source that justifies it. These parameters are locked for Phase 1 and should not be changed without a documented rationale.*

### Sample Rate
**2048 Hz** (downsampled from native 4096 Hz GWOSC delivery).
*Justification: MLGWSC-1 (Schäfer et al. 2023), AResGW (Nousi et al. 2023), Aframe (Marx et al. 2024). Covers BBH merger frequencies up to 1024 Hz Nyquist. Halves memory footprint vs. native rate.*

### PSD Estimation
- **Method:** Welch's method
- **Segment length:** 4 seconds (8192 samples at 2048 Hz)
- **Overlap:** 50% (2-second hop)
- **Window:** Hann window on each segment
- **Averaging:** Median (not mean — median is robust to glitch contamination in the off-source window)
- **Off-source window:** 64 seconds of data preceding the analysis window
- **Critical constraint:** The analysis window is NEVER included in the PSD estimate.

*Justification: Aframe (Marx et al. 2024) uses 64s for filter-building; community standard is 4s segments with 50% overlap. Median averaging confirmed by multiple LIGO PE pipelines.*

### Tapering
- **Window:** Tukey window, α = 0.25
- **Applied to:** The full 2-second buffer before FFT-based whitening
- **Effect:** α=0.25 means 12.5% of the buffer is tapered on each end, rolling smoothly to zero and eliminating the Gibbs discontinuity at FFT boundaries

*Justification: Gebhard et al. 2019 ggwd default. Eliminates spectral leakage that causes Gibbs ringing in whitened time-domain data.*

### Whitening
FFT-domain operation:
```
h_w(f) = h̃(f) / sqrt(S_n(f))
```
where S_n(f) is the median-averaged Welch PSD interpolated onto the FFT frequency grid.

**Implementation:** numpy.fft.rfft → element-wise division → numpy.fft.irfft. Written explicitly in scipy/NumPy — not via gwpy's `.whiten()` black box.

### Buffer Arithmetic (locked)
```
Fetch:     |────────── 2 s buffer (4096 samples @ 2048 Hz) ──────────|
           |← 0.5s crop →|←──── 1.0s analysis ────→|← 0.5s crop →|
           └──  discard  ─┘                          └──  discard  ─┘
```
- **Buffer fetched:** 2 seconds (4096 samples)
- **Crop per edge:** 0.5 seconds (1024 samples)
- **Analysis window:** 1.0 second (2048 samples)
- **Latency floor:** 0.5 seconds (the "future" edge crop introduces this irreducible latency)

*Justification: Aframe crops 0.5s per edge to remove whitening filter settle-in and taper roll-off regions.*

### Bandpass Filtering
- **Type:** Butterworth 4th-order, zero-phase (scipy.signal.filtfilt)
- **Passband:** 20 Hz – 1000 Hz
- **Applied to:** The final 1-second cropped whitened window (not the buffer)
- **Why post-whitening:** Whitening changes the effective frequency response; applying bandpass before whitening can create edge artifacts that survive into the analysis window

### Cross-Correlation Edge Features
Computed on the final 1-second cropped whitened window:

```python
# FFT-based cross-correlation (fast, exact)
R_ij(τ) = IFFT[conj(FFT(h_w_i)) * FFT(h_w_j)]

# Physical search window
τ_peak = argmax |R_ij(τ)| for τ ∈ [-τ_max, +τ_max]
# τ_max = 10ms for H1-L1; 27ms for H1-V1, L1-V1

# Edge feature vector
e_ij = [τ_peak, |R_ij(τ_peak)|]  ∈ ℝ^2
```

**Critical timing:** Cross-correlation is computed AFTER edge cropping. Computing on the uncropped buffer creates artificial peak at τ=0 from matching Tukey taper artifacts, masking the true astrophysical time delay.

### Preprocessing Stack Summary

| Step | Tool | Parameters |
|---|---|---|
| Data fetch | gwpy | GPS time, 64s+2s segment |
| Downsample | scipy.signal.resample | 4096 → 2048 Hz |
| PSD estimation | scipy.signal.welch | nperseg=8192, noverlap=4096, window='hann', average='median' |
| Taper | scipy.signal.get_window('tukey', N, 0.25) | α=0.25 |
| Whiten | numpy.fft.rfft/irfft | divide by sqrt(S_n(f)) |
| Crop | array slice | remove first/last 1024 samples |
| Bandpass | scipy.signal.butter + filtfilt | 4th order, 20-1000 Hz, zero-phase |
| Cross-correlation | numpy.fft + argmax | ±10ms / ±27ms search windows |

---

## 6. Literature Review Record

*This section catalogs every paper read for this project, its contribution, and how it influenced QGRAV's design.*

### Foundational Papers (Pre-GNN Era)

**George & Huerta (2018)** — *Phys. Rev. D 97, 044039*
The seminal deep learning GW paper. 1D CNN on whitened strain, 1 second at 8192 Hz, single-detector H1. Established that deep learning can match matched filtering on clean simulated data. This is QGRAV's primary baseline.

**Gabbard et al. (2018)** — *Phys. Rev. Lett. 120, 141103*
Confirmed George & Huerta's results. Used Tukey window α=1/8=0.125 for pre-whitening taper. Informed our preprocessing parameter range.

**Gebhard et al. (2019)** — *Phys. Rev. D 100, 063015*
Critical paper for preprocessing. Explicitly stated the "over-fetch, taper, crop" requirement: "δt should be chosen larger than half the desired sample length, because the discrete Fourier transform as part of a whitening procedure corrupts the edges at both ends, which need to be cropped off." Used Tukey α=0.25 (`fade_on` in ggwd). Informed our buffer arithmetic directly.

### Community Benchmark Papers

**MLGWSC-1 / Schäfer et al. (2023)** — *Phys. Rev. D 107, 023021*
The community benchmarking challenge. Established the de facto standard: 2048 Hz, 1-second analysis windows, H1+L1 parallel channels. All QGRAV parameters are anchored to this standard.

**AResGW / Nousi et al. (2023)** — *Phys. Rev. D 108, 024022*
Confirmed 2048 Hz, 1s analysis window. In real-time mode, uses 4.25s for PSD estimation; crops 0.125s per edge from 1.25s whitened segment to yield 1s output.

**Aframe / Marx et al. (2024)** — *arXiv:2403.18661*
Our primary preprocessing reference. 64s off-source PSD estimation, 2s buffer, 0.5s crop per edge (whitening filter settle-in), 1.5s analyzed (in their architecture; we use 1.0s). Explicitly states: "Due to whitening filter settle-in, 0.5s of data is corrupted on both ends of the window and removed." Directly informed our buffer arithmetic.

### Direct Competitors (Multi-Detector GNN Era)

**Tian, Huerta, Zheng & Kumar (2024)** — *Mach. Learn. Sci. Tech. 5, 025056* (ML:ST)
arXiv:2306.15728. **The closest prior work.** HDCN (WaveNet-style dilated convolution) + MPNN (message passing with max pooling) for three-detector HLV network. 1.2 million training waveforms (higher-order modes), trained on 256 A100 GPUs at Argonne. 4-model ensemble achieves 2 false positives per decade. Validated on O3b data: found 6 events in February 2020, zero false positives.

*Impact on QGRAV:* This paper validates the entire research direction and was published in ML:ST (our target journal). However, it uses MPNN with max pooling — no edge features, no attention mechanism, no event-specific detector weighting. Their "graph" is a node aggregation without any physics-informed edges. Confirmed via code inspection of `utils.py`: preprocessing is just `normalize(whitened_data)` and the graph structure is implicit in the model with `np.stack((L1, H1, V1), axis=1)`.

**AttenGW / Tiki & Huerta (2025)** — *arXiv:2512.12513*
From the same Argonne/Huerta group. HDCN + Cross-Attention Network (CAN). Cross-attention operates along the time dimension: H1 features form queries, L1 features form keys/values (and symmetric). 193K parameters, lightweight. On February 2020 O3b data: single model 124 background triggers, 3-model ensemble 0 false positives (vs Tian et al.'s 6-model ensemble). Released as open-source Python/PyTorch package.

*Impact on QGRAV:* **This paper means QGRAV cannot claim "first attention-based multi-detector GW detection."** AttenGW uses cross-attention at the temporal level (each H1 timestep attends to all L1 timesteps). QGRAV uses graph attention at the detector node level with explicit physics-informed edge features. These are architecturally and conceptually distinct. AttenGW's attention learns temporal alignment; QGRAV's attention learns event-level detector quality weighting as a function of measured cross-correlation coherence. No physics-informed edge features in AttenGW.

**Skliris, Norman & Sutton (2020/2024)** — *arXiv:2009.14611*
MLy pipeline. Two CNNs: coincidence model (whitened strain) + coherence model (whitened strain + Pearson cross-correlation between all detector pairs). Cross-correlation computed for all time delays up to ±30ms. Scores multiplied together. Achieves sensitivity approaching cWB gold standard. Targets unmodeled gravitational-wave bursts (no CBC template assumption). TensorFlow/Keras.

*Impact on QGRAV:* This paper uses cross-correlation as a CNN input — the closest prior approach to our edge feature concept. However, MLy feeds the full correlation timeseries as a separate CNN branch without graph structure. QGRAV's innovation is encoding the cross-correlation as a **graph edge feature** in a GATv2 architecture, enabling physics-informed attention. MLy has no graph, no node attention, no event-specific detector weighting.

### Quantum GW Papers

**Rodrigues de Miranda et al. (2025)** — *Quantum Machine Intelligence 7:17*
Quantum Variational Rewinding (QVR) for GW detection as anomaly detection. 2 qubits on IBM ibm\_kyoto. Simulation: perfect accuracy on filtered O3 data. Hardware: ran on 2 signals as proof of concept, results promising but extremely limited.

*Impact on QGRAV:* Establishes prior art for quantum GW detection but uses a fundamentally different algorithm (anomaly detection, not classification) and much simpler circuit (2 qubits vs our 8). QGRAV's VQC is a classification head integrated into a hybrid GNN architecture, not a standalone detector. Both quantum approaches agree: hardware noise is significant and the contribution is the benchmark, not claimed advantage.

**Isfan, Caramete & Caramete (2026)** — *arXiv:2601.14036*
Evaluated cloud quantum computers (IBM, IonQ, Azure) for QNN on LISA GW data. Critical findings:
- IBM Kyoto hardware accuracy: **18.7%** vs **100%** on simulator for 4-qubit QNN
- Full QNN training costs: $2,000 (IBM), $58,000 (Amazon Braket), $1,000,000 (Azure)
- Qiskit version incompatibilities between `qiskit` and `qiskit-ibm-runtime` are a major practical problem
- Classifier develops catastrophic bias toward class 0 on hardware

*Impact on QGRAV Phase 3:* This paper is essential for framing hardware results honestly. When QGRAV Phase 3 shows hardware performance degradation (which it will), we cite Isfan et al. 2026 and write: "Consistent with recent findings on quantum hardware degradation for QNN-based GW detection tasks, we observe significant performance reduction on IBM Quantum hardware relative to AerSimulator, confirming that hardware noise characterization is the primary contribution of Phase 3." The 18.7% result also sets realistic expectations for what to expect.

**Practical note on IBM free plan:** 10 minutes of quantum compute time per month. Only validation subset (not training) runs on hardware. All training stays on AerSimulator. This is exactly the workflow we planned.

---

## 7. Novel Contribution Statement (Current)

*This section records the precise evolution of our novelty claim. The claim below is the version that survives the full literature review.*

### Version 1 (May 7, 2026 — pre-literature review)
"The first deep learning paper to model the LIGO-Virgo detector network as a graph and use a Graph Attention Network for multi-detector gravitational wave detection."

**Status: Superseded.** Tian et al. 2024 used a GNN (MPNN) for multi-detector GW detection.

### Version 2 (After Tian et al. analysis)
"The first graph attention network for multi-detector gravitational wave detection, with learned event-specific detector weighting instead of fixed max pooling."

**Status: Partially superseded.** AttenGW (Dec 2025) uses an attention mechanism for multi-detector aggregation, though based on cross-attention not graph attention.

### Version 3 (Current — after full literature review)

> *QGRAV is the first architecture to encode the measured Pearson cross-correlation coherence and inter-detector time-delay geometry as physics-informed graph edge features in a GATv2 attention layer, enabling event-specific detector weighting as a learned function of measured network coherence.*

**What makes this claim bulletproof:**

- Tian et al. (GNN): graph structure, no edge features, max pooling — not attention, not physics-informed
- AttenGW (attention): temporal cross-attention along time dimension, no physics-informed edge features, no graph topology
- MLy (cross-correlation): Pearson correlation as CNN input, no graph structure, no attention
- QVR (quantum): anomaly detection, 2 qubits, no GNN, no cross-correlation edge features

QGRAV is the first architecture where ALL of these exist simultaneously:
1. Graph topology modeling the detector network
2. Physics-informed edge features (cross-correlation coherence + time-delay geometry)
3. GATv2 attention (dynamic, event-specific weighting)
4. Integrated VQC classifier on the graph readout

### The Four-Sentence Paper Abstract Framing

Existing multi-detector deep learning approaches either process detectors independently (George & Huerta 2018, Gabbard et al. 2018), aggregate with fixed permutation-invariant operations without physics priors (Tian et al. 2024), or use temporal cross-attention without explicit coherence features (AttenGW, Tiki & Huerta 2025). We present QGRAV, a hybrid quantum-classical graph attention network where the measured inter-detector Pearson cross-correlation lag and coherence score are encoded as physics-informed edge features in a GATv2 layer, enabling event-specific detector weighting as a learned function of measured network coherence. We evaluate QGRAV against matched filtering (PyCBC), single-detector CNN, and AttenGW baselines on real LIGO O3 data with injected CBC waveforms, using detection efficiency vs. SNR curves, ROC analysis, and false alarm rate on the GravitySpy glitch catalog. In Phase 3, we replace the classical classifier with a variational quantum circuit and benchmark detection performance under both noiseless simulation and real IBM Quantum hardware noise, providing the first systematic characterization of hardware decoherence effects on hybrid GNN+VQC gravitational wave detection.

---

## 8. Publication Strategy

### Target Journal
**Machine Learning: Science and Technology** (IOP Publishing)
- Indexed: Web of Science, Scopus, PubMed
- Open access
- Review timeline: 6–10 weeks
- This journal published Tian et al. 2024 (direct competitor) — editorial board accepts this class of work

### Backup Journal
**Astronomy and Computing** (Elsevier)
- Indexed, lower profile than ML:ST
- Natural fit for computational methods in astronomy framing

### Conference Workshop (Parallel Track)
**ML4PS at NeurIPS 2026** — Machine Learning and the Physical Sciences workshop
- 4-page extended abstract
- Phase 1 results alone are submittable
- Deadline: typically August/September
- Virtual presentation acceptable post-COVID

### Projected Timeline

| Milestone | Target Date |
|---|---|
| Environment setup + data pipeline | July 2026 |
| Phase 1 training (CNN vs GNN) | July–August 2026 |
| Phase 2 (matched filtering + ablations) | August–September 2026 |
| Phase 3 (VQC + IBM Quantum) | September–October 2026 |
| Paper written | October–November 2026 |
| arXiv preprint posted | November 2026 |
| ML:ST submission | November 2026 |
| Expected acceptance | Early 2027 |
| Published with DOI | Spring 2027 |

### What Makes It Publishable
1. **Real data** — LIGO O3 strain, not simulation. Non-negotiable for ML:ST.
2. **Matched filtering baseline** — comparative evaluation against PyCBC/GstLAL.
3. **Ablation studies** — remove GATv2 → remove V1 → classical vs VQC head. Proves each component contributes.
4. **GravitySpy glitch test** — demonstrates false alarm reduction on real non-Gaussian transients.
5. **Hardware quantum validation** — IBM Quantum noise characterization table.
6. **Precise null result framing** — if VQC doesn't beat MLP, that is a publishable benchmark result.

---

## 9. Platform and Infrastructure Decisions

### Lightning AI
- **Why:** Persistent storage (`/teamspace`) survives between sessions; conda available for PyCBC; T4 GPU on free tier; VS Code IDE
- **LIGO data:** Downloaded once, cached to `/teamspace`, never re-downloaded
- **Environment:** Installed once via `setup.sh`, never rebuilt

### Code Architecture
```
QGRAV/
├── data/
│   ├── download.py       # gwpy GWOSC fetcher (only gwpy usage)
│   ├── preprocess.py     # scipy/NumPy DSP pipeline (all explicit)
│   ├── inject.py         # PyCBC waveform injection
│   ├── dataset.py        # HDF5 PyTorch DataLoader
│   └── edge_features.py  # Cross-correlation edge extractor
├── model/
│   ├── encoder.py        # 1D CNN (shared weights)
│   ├── graph.py          # PyG Data object construction
│   ├── gat.py            # GATv2Conv (4 heads, edge_dim=2)
│   ├── classifier.py     # Classical linear head
│   └── vqc.py            # Phase 3: ZZFeatureMap + RealAmplitudes
├── training/
│   ├── module.py         # PyTorch Lightning LightningModule
│   └── train.py          # Entry point
├── evaluation/
│   ├── metrics.py        # Detection efficiency, ROC, FAR
│   ├── baseline_mf.py    # PyCBC matched filtering
│   └── ablation.py       # Ablation study runner
└── setup.sh              # One-time environment bootstrap
```

### Key Library Versions (to be locked in setup.sh)
- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric 2.5+
- Qiskit 1.0+
- qiskit-machine-learning (latest)
- qiskit-ibm-runtime (latest — watch for version conflicts, see Isfan et al. 2026)
- gwpy (latest)
- PyCBC (conda, latest)
- scipy, numpy, h5py, matplotlib

---

## 10. Open Questions and Future Work

### Unresolved Architectural Questions (for ablation study)

1. **HDCN vs 1D CNN:** AttenGW and Tian et al. use WaveNet-style dilated convolutions (HDCN). Our Phase 1 uses a simpler 1D CNN. If Phase 1 performance is disappointing, swapping the encoder to HDCN is the first lever. Reference implementation: `AttenGW/model/hdcn.py` (open source, Tiki & Huerta 2025).

2. **Max pooling vs mean pooling vs concatenation for graph readout:** Tian et al. found max pooling optimal in their ablation. QGRAV uses mean pooling as the default. This should be included in the ablation table.

3. **Number of attention heads:** 4 heads chosen as default. Whether 2, 4, or 8 heads optimize performance on this task is worth ablating.

4. **Training data class balance:** Tian et al. used 70% negative / 30% positive. AttenGW used 65% noise-only. Our default is 65% noise-only. This ratio affects both training stability and generalization.

### Phase 3 Quantum Questions

1. **Circuit depth vs hardware noise:** The 8-qubit ZZFeatureMap + RealAmplitudes circuit has depth ~16–26 depending on optimization. Given Isfan et al. 2026 showing 18.7% accuracy at 4-qubit depth ~13, our 8-qubit circuit is likely to perform comparably poorly on hardware. This is expected and should be reported as the characterization result.

2. **Qubit count sensitivity:** Should we try 4 qubits (matching Isfan et al.) vs 8 qubits? A 4-qubit ablation might be informative.

3. **VQC training instability (barren plateaus):** Rodrigues de Miranda et al. observed training instability consistent with barren plateau phenomenon. With 8 qubits and 3 repetitions, this may be a practical challenge. COBYLA (gradient-free) optimizer may be more stable than Adam for the quantum parameters.

### Publication Questions

1. **Co-authorship:** Dr. Zhiqian Chen (MSU, GNN expertise) has not been contacted yet. This remains the highest-leverage non-code action in the project. A faculty co-author strengthens the paper's credibility and the GNN methodology framing substantially.

2. **AttenGW authors as reviewers:** Tiki & Huerta are from Argonne, connected to the Huerta group that reviews this area of ML:ST regularly. QGRAV must be clearly differentiated from AttenGW in the cover letter and abstract — the distinction is physics-informed graph edge features vs temporal cross-attention.

---

*Document last updated: June 2026*
*Next planned update: After data pipeline is complete and first training results are in*

---

> *"The purpose of this document is not to justify the decisions we made but to record them honestly, including the ones that were later superseded. A research record that only shows the final path is a fiction. This document shows the actual path."*
