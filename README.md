# QGRAV: Hybrid Quantum-Classical Graph Attention Network for Gravitational Wave Detection

<div align="center">

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![PyG](https://img.shields.io/badge/PyG-2.5%2B-3c9cd7)
![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-6929C4)
![License](https://img.shields.io/badge/license-MIT-green)
![Target](https://img.shields.io/badge/target-ML%3AST%20%7C%20IOP-lightgrey)

**Adi Singh** · Mississippi State University  

</div>

---

## Abstract

We present QGRAV, a hybrid quantum-classical deep learning framework for detecting gravitational wave (GW) signals in real multi-detector LIGO-Virgo strain data. Existing deep learning approaches process each interferometer independently, discarding the inter-detector network coherence that physically distinguishes astrophysical signals from instrumental noise transients. QGRAV addresses this fundamental limitation by modeling the LIGO-Virgo observatory network as a directed graph, where detectors are nodes carrying CNN-extracted feature vectors and edges encode the measured cross-correlation lag and coherence score between detector pairs — a physics-informed geometric prior derived directly from the speed-of-light propagation constraint. A four-head GATv2 attention layer learns to weight inter-detector correlations end-to-end, with attention weights that are interpretable as data-driven detector quality metrics. As an exploratory extension, the classical classification head is replaced with a variational quantum circuit (VQC) trained on Qiskit's AerSimulator and validated on real IBM Quantum hardware, providing the first rigorous benchmark of quantum classifiers on real gravitational wave detection data. QGRAV is evaluated on LIGO O3 strain data against matched filtering (PyCBC), single-detector CNN, and ablation baselines using detection efficiency vs. optimal SNR curves, ROC analysis, and false alarm rate on the GravitySpy glitch catalog.

---

## Motivation: The Multi-Detector Coherence Gap

Since the first deep learning paper on GW detection (George & Huerta 2018), the standard paradigm has been to run a neural network independently on each interferometer's whitened strain and fuse results through concatenation or majority voting. This design has a fundamental physical flaw: **it treats the detector network as a bag of independent sensors rather than as a geometrically constrained array**.

A real gravitational wave signal is coherent across the network. The H1–L1 propagation delay is bounded by the light-travel time between Hanford, WA and Livingston, LA (~10 ms maximum). H1–Virgo and L1–Virgo delays are bounded by ~27 ms. An astrophysical signal must arrive at each detector with a time delay consistent with a specific sky position. A non-Gaussian noise transient (glitch) — the dominant source of false alarms in matched filtering pipelines — appears in at most one detector with no coherent counterpart in the others.

Current deep learning models discard this constraint entirely. **QGRAV builds it in as an architectural prior.**

The inter-detector cross-correlation at the observed lag encodes both the signal's sky geometry (via $\tau_\text{peak}$) and its network coherence (via $|R_{ij}(\tau_\text{peak})|$). By encoding these as graph edge features and training a graph attention network to weight them, QGRAV learns glitch rejection not from labeled glitch data, but as an emergent property of the physics-informed graph structure.

---

## Novel Contributions

1. **Physics-informed graph representation of the detector network.** The first architecture to model the LIGO-Virgo network as a graph where edge features are derived from the measured inter-detector cross-correlation, encoding both time-delay geometry and signal coherence as a trainable architectural prior.

2. **Emergent glitch rejection via attention.** High-coherence events drive GAT attention toward cross-detector aggregation; single-detector glitches drive attention toward self-loops — producing false alarm suppression without explicit glitch labels, demonstrated against the GravitySpy non-Gaussian transient catalog.

3. **Sub-millisecond inference latency.** QGRAV enables real-time detection triggering for electromagnetic follow-up networks (e.g., GOTO, BlackGEM) in the binary neutron star merger scenario, where the kilonova blue-component peaks within hours of merger.

4. **First rigorous VQC benchmark on real gravitational wave data.** A variational quantum circuit classifier trained on AerSimulator and validated on IBM Quantum hardware provides hardware noise characterization for this detection task — a publishable null-result benchmark regardless of quantum vs. classical performance outcome.

---

## Architecture

```
H1 strain ──► Preprocessing ──► CNN Encoder (shared weights) ──► h_H1 ∈ ℝ^256 ──┐
                                                                                    │
L1 strain ──► Preprocessing ──► CNN Encoder (shared weights) ──► h_L1 ∈ ℝ^256 ──┤
                                                                                    │
            Cross-correlation edge features:                                        │
            e_H1L1 = [τ_peak, |R(τ_peak)|] ∈ ℝ^2  ──────────────────────────────┤
                                                                                    ▼
                                                          Graph construction
                                                  (nodes = detectors, edges = Δt)
                                                                    │
                                                                    ▼
                                                    GATv2 Layer (4-head attention)
                                                  α_ij = softmax(a^T LeakyReLU(W[h_i‖h_j‖e_ij]))
                                                                    │
                                              ┌─────────────────────┴──────────────────────┐
                                         Phase 1/2                                    Phase 3
                                    Classical Head                               Quantum Head
                               nn.Linear(256→1) → σ                    ZZFeatureMap(8q) + RealAmplitudes
                                         └─────────────────────┬──────────────────────┘
                                                               ▼
                                                      p(GW signal) ∈ [0,1]
```

### Preprocessing Pipeline

All preprocessing follows the MLGWSC-1 / Aframe community standard, justified from the literature:

| Step | Parameters | Justification |
|---|---|---|
| Downsample | 4096 → 2048 Hz | Gebhard et al. 2019; covers BBH merger frequencies to 1024 Hz Nyquist |
| PSD estimation | Welch, 4 s segments, 50% overlap, Hann window, median averaging, 64 s off-source window | Aframe (Marx et al. 2024); stable low-variance PSD down to 20 Hz |
| Tapering | Tukey window, α = 0.25, applied to 2 s buffer | Gebhard et al. 2019; eliminates Gibbs discontinuity at FFT boundaries |
| Whitening | FFT-domain: $\tilde{h}_w(f) = \tilde{h}(f) / \sqrt{S_n(f)}$ | Standard GW data analysis; equalizes frequency-band contributions |
| Edge crop | 0.5 s per edge after whitening | Aframe (Marx et al. 2024); removes taper roll-off and filter settle-in |
| Bandpass | Butterworth 4th-order, 20–1000 Hz, zero-phase (`filtfilt`) | Applied post-whitening on the cropped 1 s window |
| Analysis window | 1.0 s (2048 samples) per detector | AResGW (Nousi et al. 2023); consistent with MLGWSC-1 |

**Buffer arithmetic:**
```
Fetch:  |────────── 2 s buffer (4096 samples) ──────────|
After taper + whiten:
        |─ 0.5 s crop ─|── 1.0 s analysis ──|─ 0.5 s crop ─|
        └──  discard  ──┘                    └──  discard  ──┘
```

### Cross-Correlation Edge Features

Edge features are computed on the final cropped 1 s whitened window — not on the uncropped buffer, which would introduce artificial correlation from symmetric taper artifacts:

$$R_{ij}(\tau) = \mathcal{F}^{-1}\!\left[\tilde{h}_{w,i}^*(f) \cdot \tilde{h}_{w,j}(f)\right]$$

$$\mathbf{e}_{ij} = \left[\tau_\text{peak},\; |R_{ij}(\tau_\text{peak})|\right] \in \mathbb{R}^2, \quad \tau_\text{peak} \in [-\Delta t_\text{max},\; +\Delta t_\text{max}]$$

where $\Delta t_\text{max} \approx 10\,\text{ms}$ for H1–L1 and $\approx 27\,\text{ms}$ for H1–V1 and L1–V1. The peak coherence $|R_{ij}(\tau_\text{peak})|$ is the primary glitch rejection signal: high for coherent astrophysical events, near-zero for single-detector transients.

### CNN Encoder

A 1D convolutional encoder with shared weights across all detectors maps each 2048-sample whitened strain vector to a 256-dimensional feature vector. Shared weights enforce the physical symmetry that the same signal morphology (a CBC chirp) should produce the same representation regardless of which detector observed it. Architecture: three convolutional blocks (Conv1D → BatchNorm → ReLU → MaxPool), followed by a linear projection to 256 dimensions.

### Graph Attention Layer (GATv2)

We use GATv2Conv (Brody et al. 2022) over the original GAT formulation (Veličković et al. 2018) for its strictly more expressive dynamic attention mechanism. For each attention head $k$ and node pair $(i, j)$:

$$e_{ij}^{(k)} = \mathbf{a}^{(k)T}\,\text{LeakyReLU}\!\left(\mathbf{W}^{(k)}\left[\mathbf{h}_i \;\|\; \mathbf{h}_j \;\|\; \mathbf{W}_e^{(k)}\mathbf{e}_{ij}\right]\right)$$

$$\alpha_{ij}^{(k)} = \text{softmax}_{j \in \mathcal{N}(i) \cup \{i\}}\!\left(e_{ij}^{(k)}\right)$$

$$\mathbf{h}_i' = \left\|_{k=1}^{4}\; \sigma\!\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij}^{(k)}\, \mathbf{W}^{(k)}\mathbf{h}_j\right)\right.$$

Self-loops are required: with two nodes and no self-loops, softmax over a single neighbor is always 1.0, eliminating the learned weighting. With self-loops, each head learns the competition between $\alpha_{ii}$ (self-attention, dominates for glitches) and $\alpha_{ij}$ (cross-attention, dominates for coherent signals). Four heads concatenate to preserve the 256-dim representation ($4 \times 64$).

### Quantum Classifier Head (Phase 3)

The quantum head replaces the classical linear classifier. A linear bridge layer $\mathbf{W}_b \in \mathbb{R}^{8 \times 256}$ compresses the graph readout to 8 dimensions for qubit encoding. The VQC consists of:

- **ZZFeatureMap** (8 qubits, 2 repetitions): encodes classical features as qubit rotation angles with entangling ZZ interactions
- **RealAmplitudes** (8 qubits, 3 repetitions): variational ansatz of $R_y$ rotations and CNOT ladders with trainable parameters
- **TorchConnector** (Qiskit Machine Learning): wraps the full circuit as `nn.Module`, training via the parameter-shift rule

Training is performed exclusively on `AerSimulator` (noiseless). A held-out validation subset is submitted to real IBM Quantum hardware via `qiskit-ibm-runtime` to characterize the effect of hardware noise on detection efficiency — providing the simulation vs. hardware comparison table.

---

## Three-Phase Research Plan

### Phase 1 — Core Claim Validation *(current)*
**Goal:** Prove that GNN > single-detector CNN on detection efficiency at matched SNR.

- 1D CNN encoder with shared weights
- H1 + L1 two-detector graph (Phase 1 only)
- GATv2, 4-head attention, cross-correlation edge features
- Classical linear classifier head
- Baseline: single-detector CNN on H1 only

**Exit criterion:** Statistically significant improvement in detection efficiency at SNR = 8 over single-detector baseline. Phase 1 results alone are submittable.

### Phase 2 — Complete Scientific Paper *(after Phase 1)*
**Goal:** Journal-ready evaluation against all relevant baselines.

- Matched filtering baseline (PyCBC)
- Virgo (V1) added as third node; ablation test of two- vs. three-detector performance
- GravitySpy glitch rejection test (FAR on non-Gaussian transients)
- Full ROC curves and detection efficiency vs. optimal SNR curves
- Ablation table: remove GAT → remove V1 → classical vs. quantum head

### Phase 3 — Quantum Extension *(after Phase 2)*
**Goal:** First rigorous VQC benchmark on real gravitational wave data.

> *"As an exploratory extension, we replace the classical classification head with a variational quantum circuit and benchmark detection performance under both noiseless simulation and real IBM Quantum hardware noise, investigating whether quantum feature maps improve sensitivity at the low-SNR detection threshold."*

- VQC head (Qiskit ZZFeatureMap + RealAmplitudes, TorchConnector)
- AerSimulator training
- IBM Quantum hardware validation (free tier, held-out test subset)
- Simulation vs. hardware noise characterization table

---

## Dataset

**Source:** LIGO Open Science Center (GWOSC), O3 Observing Run (April 2019 – March 2020)  
**Access:** [`gwpy.timeseries.TimeSeries.fetch_open_data()`](https://gwpy.github.io/)  
**Storage:** Downloaded once to `/teamspace` persistent storage on Lightning AI

**Training data construction:**

| Class | Source | Label |
|---|---|---|
| Noise | Real O3 background strain segments, H1 + L1 | 0 |
| Signal | Synthetic CBC waveforms (PyCBC) injected into real noise at SNR ∈ [5, 30] | 1 |
| Glitch | GravitySpy labeled non-Gaussian transients (Phase 2 only) | 0 (hard negative) |

**Injection protocol:**  
- Waveform parameters: component masses $m_1, m_2 \in [5, 80]\,M_\odot$ (BBH), uniform in chirp mass
- Merger epoch: uniform random within central 0.5 s of the 1 s analysis window
- Sky position: isotropic; inter-detector time delays computed from sky localization and applied during injection
- SNR definition: optimal SNR in H1, matched-filter definition

**Test set:** Confirmed O3 events from the GWTC-3 catalog used for real-event validation

---

## Installation

> **Platform:** Lightning AI (persistent storage, T4 GPU, conda available). See [`setup.sh`](setup.sh).

```bash
# 1. Clone the repository
git clone https://github.com/operator2036/QGRAV.git
cd QGRAV

# 2. PyTorch Geometric (order-sensitive install)
pip install torch-scatter torch-sparse torch-geometric \
  -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html

# 3. GW data stack (PyCBC requires conda for LALSuite C/Fortran dependencies)
conda install -c conda-forge pycbc gwpy

# 4. Quantum stack
pip install qiskit qiskit-machine-learning qiskit-ibm-runtime qiskit-aer

# 5. Training and utilities
pip install lightning h5py scipy numpy matplotlib
```

---

## Evaluation Protocol

The primary evaluation follows the LIGO detection community standard:

| Metric | Description |
|---|---|
| Detection efficiency vs. optimal SNR | Fraction of injections recovered above threshold as a function of SNR; primary result figure |
| Detection efficiency at SNR = 8 | Single-number comparison at the LIGO operational threshold |
| ROC curve + AUC | Receiver operating characteristic across all classifiers on the same axis |
| FAR at fixed threshold | False alarm rate per hour at the operating threshold; evaluated on GravitySpy glitches |
| Inference latency | Wall-clock time from input strain to detection decision |

---

## Results

*In progress. Phase 1 training underway.*

| Method | Eff. @ SNR=8 | FAR/hr | AUC | Latency |
|---|---|---|---|---|
| Matched filtering (PyCBC) | — | — | — | ~seconds |
| Single-detector CNN (H1 only) | — | — | — | <1 ms |
| QGRAV — GNN, classical head | — | — | — | <1 ms |
| QGRAV — GNN, VQC head (simulated) | — | — | — | — |
| QGRAV — GNN, VQC head (IBM hardware) | — | — | — | — |

---

## Repository Structure

```
QGRAV/
├── data/
│   ├── download.py          # GWOSC strain fetcher (gwpy)
│   ├── preprocess.py        # Bandpass, whiten, taper, crop pipeline
│   ├── inject.py            # CBC waveform injection (PyCBC)
│   ├── dataset.py           # HDF5 dataset loader (PyTorch)
│   └── edge_features.py     # Cross-correlation edge feature extractor
├── model/
│   ├── encoder.py           # 1D CNN encoder (shared weights)
│   ├── graph.py             # PyTorch Geometric graph construction
│   ├── gat.py               # GATv2 4-head attention layer
│   ├── classifier.py        # Classical linear head
│   └── vqc.py               # Variational quantum circuit head (Phase 3)
├── training/
│   ├── module.py            # PyTorch Lightning LightningModule
│   └── train.py             # Training entry point
├── evaluation/
│   ├── metrics.py           # Detection efficiency, ROC, FAR computation
│   ├── baseline_mf.py       # Matched filtering baseline (PyCBC)
│   └── ablation.py          # Ablation study runner
├── notebooks/
│   └── exploration/         # Data visualization and sanity checks
├── configs/
│   └── phase1.yaml          # Hyperparameter configuration
├── setup.sh                 # Environment bootstrap script
└── README.md
```

---

## Citation

If you use QGRAV in your research, please cite:

```bibtex
@article{singh2026qgrav,
  title   = {{QGRAV}: Hybrid Quantum-Classical Graph Attention Network for Gravitational Wave Detection},
  author  = {Singh, Adi},
  journal = {Machine Learning: Science and Technology},
  year    = {2026},
  note    = {In preparation}
}
```

---

## Author

**Adi Singh**  
B.S. Computer Science, Mississippi State University  
M.S. Computer Science, Mississippi State University
GitHub: [@researchingadi](https://github.com/researchingadi

---

## Acknowledgments

This work uses data from the LIGO Open Science Center (GWOSC). LIGO is funded by the NSF and operated by Caltech and MIT. The authors thank the LIGO Scientific Collaboration and the Virgo Collaboration for making O3 data publicly available.

Gravitational wave strain data: [gwosc.org](https://gwosc.org)  
MLGWSC-1 benchmark: Schäfer et al., *Phys. Rev. D* 107, 023021 (2023)  
Aframe pipeline: Marx et al., *arXiv:2403.18661* (2024)

---

## License

MIT License. See [`LICENSE`](LICENSE) for details.

---

<div align="center">
<sub>Built with real LIGO data. Targeting ML:Science and Technology (IOP Publishing).</sub>
</div>
