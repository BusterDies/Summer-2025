# Summer-2025 Dementia Multimodal Pipeline

## Overview

This repository contains a multimodal data-processing pipeline used to build machine-learning datasets for dementia-symptom detection from video recordings.

The pipeline extracts three synchronized modalities from each video:

* Facial Action Units (OpenFace-based)
* Vocal prosody features (OpenSMILE eGeMAPS)
* Time-aligned speech transcripts (WhisperX)

These outputs are combined into Label Studio tasks, annotated, and converted into training datasets.

---

## Pipeline Structure

### Stage 1 — Feature Extraction

Located in the `pipeline/` directory.

Core scripts:

* run_pipeline.py
* extract_au10.py
* extract_prosody4.py
* whisperxdi3.py

Run the full pipeline on a video:

python pipeline/run_pipeline.py <video.mp4>

This produces files inside:

processeddata/task-1/

including:

* video.mp4
* *_au.csv
* *_prosody.csv
* *_transcript.json
* task-#.json

The generated task JSON file is imported into Label Studio.

---

### Stage 2 — Annotation

Annotations are created in Label Studio using:

* video playback
* AU CSV
* prosody CSV
* transcript JSON

Label Studio exports are stored locally and are not tracked in git.

---

### Stage 3 — Dataset Construction

Sliding-window dataset builder:

* pipeline/build_dataset_v3.py
* pipeline/ls_loader_v2.py

Build dataset:

python pipeline/build_dataset_v3.py

Train baseline model:

python pipeline/build_dataset_v3.py --train

Add temporal context:

python pipeline/build_dataset_v3.py --history 3

Outputs are saved in:

datasets_movies_v2/

including:

* X.npy
* Y.npy
* groups.npy
* scaler.joblib

---

## Modalities

### Facial Features

Extracted using pipeline/extract_au10.py

Output:
*_au.csv

Action Units:
AU1, AU2, AU4, AU6, AU9, AU12, AU25, AU26

---

### Vocal Prosody

Extracted using pipeline/extract_prosody4.py

Output:
*_prosody.csv

Feature dimension:
88 low-level descriptors (eGeMAPS)

---

### Transcript

Generated using pipeline/whisperxdi3.py

Output:
*_transcript.json

Each segment contains:
start, end, text

Speaker labels are currently not used by the dataset builder.

---

## Dataset Format

Each training sample contains:

[ AU_mean | Prosody_mean | Text_embedding ]

Text embeddings are generated using:

SentenceTransformer("all-MiniLM-L6-v2")

Labels:

* mood_dysregulation
* problems_communicating
* confusion_disorientation
* memory_loss

---

## Repository Layout

pipeline/ — core pipeline scripts
modules/ — reusable feature extraction modules
slurmtesting/ — cluster job scripts
src/ — experimental utilities
Tools/ — legacy Whisper scripts

Ignored directories (large data or generated artifacts):

* DementiaBank/
* speechcorpus2020/
* processeddata/
* datasets/
* datasets_movies/
* datasets_movies_v2/
* models/
* annotations/
* annotations_movies/
* opensmile/

---

## Dependencies

Required Python packages include:

* torch
* whisperx
* sentence-transformers
* pandas
* numpy
* scikit-learn
* opensmile

Example install:

pip install whisperx sentence-transformers pandas numpy scikit-learn opensmile

---

## Design Philosophy

The pipeline follows this structure:

video → feature extraction → annotation → dataset construction → training

Artifact generation and dataset construction are separated to keep the workflow reproducible and modular.
