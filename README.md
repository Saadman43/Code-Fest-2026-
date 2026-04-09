#  EEG-Based Neuro-Symbolic Brain-Computer Interface

An end-to-end **neuro-symbolic deep learning** pipeline for decoding **overt and imagined speech** from EEG signals, designed for robot control via brain-computer interface (BCI).

---

##  Overview

This project classifies EEG signals recorded during **spoken (overt)** and **imagined (silent)** speech interactions with a simulated robot. It combines a neural EEG encoder with a **learnable symbolic reasoning layer** that enforces logical constraints (e.g., don't repeat the same command, don't navigate to a blocked direction) directly within the training loop.

The model is first trained on overt speech EEG, then **transferred** to imagined speech — a challenging paradigm shift common in real-world BCI deployment.

---

##  Architecture

```
EEG Signal (channels × time)
        │
        ▼
  ┌──────────────┐
  │  EEGNet-style │   Temporal conv → Depthwise conv → Separable conv
  │   Encoder    │
  └──────┬───────┘
         │ Neural embeddings
         ▼
  ┌─────────────────────────────┐
  │  Learnable Neuro-Symbolic   │   Differentiable symbolic rules:
  │        Layer                │   • Invalid-action penalty
  │                             │   • Repeat-command penalty
  │  + Symbolic Context Tensors │   • Feasible-action boost
  │  (blocked dirs, prev cmd,   │   • Soft gating via learned weights
  │   can_pick, can_push, ...)  │
  └──────────────┬──────────────┘
                 │
                 ▼
        Final Prediction
     (left / right / up / pick / push)
```

**Total loss = Cross-Entropy Loss + λ × Symbolic Constraint Loss**

Both losses are backpropagated jointly, making symbolic rules *trainable*, not hard-coded.

---

##  Dataset

- **Format:** MNE-compatible `.fif` files, one per participant
- **Events:** Overt and silent speech for 5 robot commands
- **Commands:** `left`, `right`, `up`, `pick`, `push`
- **Paradigms:** `overt` (spoken aloud) and `imagined` (silent)
- **Sidecar files:** `event_names.json` (event dictionary), optional `montage.bvef`

> Update `CONFIG["data_root"]` in the notebook to point to your local data directory.

---

##  Configuration

All key hyperparameters are centralized in the `CONFIG` dictionary:

| Parameter | Default | Description |
|---|---|---|
| `tmin` / `tmax` | `0.0` / `2.0` s | Epoch window |
| `l_freq` / `h_freq` | `1.0` / `40.0` Hz | Bandpass filter |
| `resample_sfreq` | `128` Hz | Target sampling rate |
| `batch_size` | `8` | Training batch size |
| `epochs_overt` | `10` | Overt training epochs |
| `epochs_imagined` | `25` | Imagined transfer epochs |
| `lr_overt` | `1e-3` | Learning rate (overt) |
| `lr_imagined` | `3e-4` | Learning rate (imagined transfer) |
| `dropout` | `0.5` | Dropout rate |
| `symbolic_min_confidence` | `0.35` | Min confidence to fire symbolic rules |
| `constraint_loss_weight` | `0.20` | Weight of symbolic constraint loss |
| `freeze_feature_extractor_during_transfer` | `False` | Freeze encoder during transfer |

---

##  Pipeline

### Stage 1 — Overt Speech Training
1. Load and preprocess `.fif` files with MNE (bandpass filter, CAR reference, epoch extraction)
2. Build symbolic context tensors from metadata (blocked directions, previous command, paradigm flag)
3. Train `EndToEndNeuroSymbolicEEGNet` on overt EEG with joint neural + symbolic loss
4. Evaluate with classification report and confusion matrix (neural-only vs. neuro-symbolic)

### Stage 2 — Transfer to Imagined Speech
5. Initialize transfer model from overt model weights
6. Fine-tune on imagined EEG at a lower learning rate
7. Evaluate and generate per-sample **explanation traces** from the symbolic layer

### Explanation Traces
Each prediction produces a human-readable trace showing:
- Neural probabilities per class
- Applied symbolic rule penalties/boosts
- Final adjusted probabilities
- Whether the symbolic layer changed the decision

---

##  Dependencies

```bash
pip install torch numpy pandas matplotlib scikit-learn mne scipy
```

| Library | Purpose |
|---|---|
| `torch` | Neural network training |
| `mne` | EEG data loading and preprocessing |
| `numpy` / `pandas` | Data manipulation |
| `matplotlib` | ERP plots, confusion matrices, training curves |
| `scikit-learn` | Metrics, label encoding, train/test splits |
| `scipy` | Optional `.mat` file support |

Python 3.8+ recommended. GPU (CUDA or Intel XPU) supported automatically.

---

##  Getting Started

1. **Clone the repository**
   ```bash
   git clone [https://github.com/Saadman43/Code-Fest-2026-)
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your data path** — open the notebook and update:


4. **Run the notebook**
   ```bash
   jupyter notebook eeg_neurosymbolic_bci.ipynb
   ```

---

##  Model Zoo (Backbone Comparison)

The notebook includes a benchmark cell comparing multiple EEG deep learning backbones:

- **EEGNet** — depthwise + separable convolutions
- **ShallowConvNet** — shallow temporal-spatial convolutions
- *(Extendable with additional architectures)*

Benchmarks run for 5 epochs each and report accuracy, balanced accuracy, and F1-macro.

---

##  Key Design Choices

- **End-to-end symbolic learning:** Symbolic rules are *not* post-hoc corrections — they are differentiable parameters trained via backpropagation alongside the neural encoder.
- **Memory-safe data loading:** Two-pass loading strategy — lightweight scan first, then per-subject epoch extraction with RAM caps and disk-backed memmaps for large datasets.
- **Reproducibility:** Seed set globally for Python, NumPy, and PyTorch (CUDA/XPU included).
- **Class imbalance handling:** Weighted random sampler + label-smoothed cross-entropy loss.
- **Cross-device support:** Automatically detects and uses CUDA GPU, Intel XPU, or CPU.

---

##  File Structure

```
.
├── eeg_neurosymbolic_bci.ipynb   
├── README.md
└── data/                         
    ├── event_names.json
    ├── montage.bvef              
    └── participant_*/
        └── *.fif
```

---

##  License

This project is released for research and academic use. Please cite appropriately if you build on this work.

---

##  Acknowledgements
- Neuro-symbolic learning framework designed for BCI explainability and constraint satisfaction
- EEG data recorded during spoken and imagined speech interaction with a simulated robot
