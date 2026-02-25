# InstruTag

A multi-label instrument tagger that listens and tell what instruments involved in a music clip: one model holds the groove (BiGRU + Attention), one model takes crisp solos (per-class XGBoost), and the ensemble lands the changes with a learned blend and per-instrument thresholds.

---

## Liner Notes

**InstruTag** predicts which instruments are present in audio. Built around **OpenMIC-2018**-style 10-second clips, it can also tag longer tracks via sliding windows.

Pipeline:
- Audio -> **VGGish embeddings**
- Probabilities from:
  - **BiGRU + Attention** (sequence model)
  - **XGBoost per class** (tabular classifier over flattened embeddings)
- Ensemble:
  - `p_ens = α·p_bigru + (1-α)·p_xgb`
- Final tags:
  - `pred = (p_ens >= thr[class])`

---

## The Band

- **BiGRU + Attention**: captures phrasing and temporal texture in embedding sequences
- **XGBoost (one model per instrument)**: sharp, class-specific boundaries
- **Ensemble**: learned `α` balances both, then thresholds turn probabilities into tags

Artifacts used at inference time:
- `outputs/bigru_attn_best.pt`
- `outputs/xgb_models_per_class.pkl`
- `outputs/scaler_pooled.pkl`
- `outputs/ensemble_alpha.npy`
- `outputs/thr_ensemble.npy`

---

## Setlist

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Backstage Pass: Download Required Files](#backstage-pass-download-required-files)
- [How to Run](#how-to-run)
  - [Clip Mode (10s)](#clip-mode-10s)
  - [Full Track Mode (sliding windows)](#full-track-mode-sliding-windows)
- [Project Layout](#project-layout)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Quick Start

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd InstruTag

python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
