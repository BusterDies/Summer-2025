# Multimodal Dementia-Emotion Classifier — Design Brief (v0.1)

## High-Level Goal
Create a **working prototype in 10 weeks** that predicts  
1. **Emotional state** – {calm, agitated, confused, anxious, neutral, happy}  
2. **Dementia symptom** – {memory-loss, disorientation, repetition, mood-swings}  

from **speech** (acoustics + text).  Face-AUs & Jetson optimisation follow later.

---

## Hardware / OS
| Device | Role | Notes |
|--------|------|-------|
| **Raspberry Pi 4** | Data logger, live demo | Python 3.11.2 |
| **Lab Server** (AMD EPYC + RTX/L40S) | Training | Ubuntu 22.04, CUDA 12 |
| **Jetson Nano Orin** | Edge test (week 8) | TensorRT ≥ 8.6 |

---

## Minimal SW Stack

| Layer | Library | Version hints |
|-------|---------|---------------|
| Acoustic features | **openSMILE 3.x** (eGeMAPS) | CLI, no Python needed |
| Prosody extras | **Parselmouth** (Praat API) | only if openSMILE jitter/shimmer insufficient |
| Offline ASR | **Vosk** small English | pip install vosk |
| Text embedding | **sentence-transformers/all-mpnet-base-v2** | 768-D |
| ML baselines | **scikit-learn**, **xgboost** | SVM, RF, XGB |
| DL models | **PyTorch 2 + Lightning** | LSTM, fusion MLP |
| Edge | **ONNX → TensorRT** | `trtexec` |

---

## File/CLI Conventions

```

data/
├─ wav/                     # raw 16 kHz mono WAV
├─ feats/
│   ├─ <utt>.csv            # openSMILE eGeMAPS (frame or functional)
│   └─ <utt>\_txt.json       # Vosk transcript + word timestamps
models/
├─ baseline\_svm.pkl
└─ lstm\_fusion.onnx

````

### Extract features (batch)
```bash
python extract_smile.py   --in wav/ --out feats/
python extract_txt.py     --in wav/ --out feats/
python extract_text_feats.py --json feats/*_txt.json --out feats/text_emb.npy
````

### Train baseline

```bash
python train_baseline.py \
  --acoustic feats/*.csv \
  --text     feats/text_emb.npy \
  --labels   labels.csv \
  --out      models/baseline_svm.pkl
```

---

## Real-Time Demo Skeleton  (Pi)

```python
# live_demo.py  (flush every 0.5 s)
import sounddevice as sd, opensmile, joblib, vosk, json
smile = opensmile.Smile(config='eGeMAPSv02')
clf   = joblib.load('models/baseline_svm.pkl')
BUFF  = 0.5    # seconds

def callback(indata, frames, time, status):
    feat = smile.process_signal(indata[:,0], 44100).iloc[-1].values
    pred = clf.predict(feat.reshape(1,-1))[0]
    print(pred)

with sd.InputStream(channels=1, samplerate=44100,
                    blocksize=int(44100*BUFF), callback=callback):
    input('press Enter to stop\n')
```

---

## Coding Standards

* **Black** + **isort**; PEP-8 line ≤ 88
* Type-hints everywhere (`mypy --strict`)
* One YAML config per experiment (`conf/exp001.yaml`)
* Log metrics via **TensorBoard**
* All scripts runnable headless (`argparse`)

---

## Evaluation Metrics

* **Emotion** – macro-F1, ROC-AUC (IEMOCAP validation)
* **Dementia** – macro-F1, Precision\@2 (ADReSS-o)
* Track latency (ms) & RAM on Pi + Jetson.

---

## 10-Week Milestones (condensed)

1️⃣ openSMILE & Vosk extractors → CSV/JSON
2️⃣ Baseline SVM/XGB on eGeMAPS+ BERT → val-F1 reported
3️⃣ Normalisation study for jitter/shimmer → lock best variant
4️⃣ Bi-LSTM seq model on acoustic frames
5️⃣ Late-fusion (LSTM + BERT) → ≥ 10 % F1 gain
6️⃣ Export ONNX, run on Jetson ≤ 200 ms
7️⃣ Live Pi demo streaming emotions every 1 s
8️⃣ Slides + JSON logs + code freeze

---

## Open Questions

* Optimal segment length for LSTM (→ experiment)
* Whether BERT on full transcript or sliding context windows gives better lift
* Speaker-level vs corpus-level z-norm for prosody features
* Data augmentation: noise, speed-perturbation?

```

---  
**Drop the block above into `design-brief.md`** – it is all Copilot needs to start generating consistent code for you.
```
