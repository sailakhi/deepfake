# AI-Driven Deepfake Detection and Digital Media Forensics

## Objective 1

Develop an AI-based deepfake detection system for manipulated facial content in images and videos using deep learning.

## System architecture

```text
Image / Video
  -> input validation
  -> image read / bounded video-frame sampling
  -> Haar face detection and largest-face crop
  -> resize + ImageNet normalization
  -> MobileNetV3-Small binary classifier
  -> P(REAL), P(FAKE), class decision
  -> video mean-P(FAKE) aggregation (when applicable)
  -> CSV/JSON forensic result
```

## Planned baseline

The baseline uses MobileNetV3-Small transfer learning for binary face classification:

- `REAL = 0`
- `FAKE = 1`

It is selected for a practical CPU-friendly final-year-project demonstration. Its ImageNet initialization is used only when the training command is run with pretrained weights enabled; the weight provenance is recorded in experiment metadata.

## Dataset and leakage prevention

Recommended practical source: FaceForensics++ video data (subject to its access terms), beginning with a documented controlled subset that contains original/real and manipulated/fake videos. Do not mix a video's frames across splits. The repository does not download any dataset automatically.

Expected source layout:

Place source media in class directories:

```text
data/raw/
  real/
    video_or_image_001.mp4
  fake/
    video_or_image_002.mp4
```

The dataset-preparation command performs a 70/15/15 stratified split of original media records before frame extraction. Consequently, frames produced from one video inherit exactly one split and cannot leak into validation or test data. A controlled subset is supported by supplying only the selected original media files under `data/raw/`; its generated counts are never hard-coded.

Generated metadata includes the sample ID, source ID, class, numeric label, split, path, and media type. `source_id` is checked so it can occur in only one split.

## Installation

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The project automatically uses CUDA when a CUDA-enabled PyTorch installation is present; the current development environment is CPU-only.

## Project structure

```text
configs/                 Runtime defaults
data/raw/                User-supplied real/fake original media
data/metadata/, splits/  Generated source-level metadata and split files
data/processed/          Generated face crops/index (ignored by Git)
src/data/                Discovery and leakage-safe splitting
src/preprocessing/       Face and frame processing
src/models/              MobileNetV3-Small baseline
src/training/            Training and checkpointing
src/evaluation/          Metrics and graphs
src/inference/           Image and video commands
src/forensic/            JSON result schema
tests/                   Deterministic unit tests
```

## Commands

Run every command from the project root using the virtual environment.

### Prepare data

```powershell
.\.venv\Scripts\python.exe -m src.data.prepare --raw-dir data/raw --metadata-path data/metadata/samples.csv --splits-dir data/splits --seed 42
.\.venv\Scripts\python.exe -m src.preprocessing.extract_faces --metadata data/metadata/samples.csv --output-dir data/processed/faces --index-path data/processed/faces.csv --every-n-frames 15 --max-frames-per-video 64
```

Inspect the actual generated split table in `data/splits/split_report.csv`; do not copy expected counts into reports before running it.

### Train

```powershell
.\.venv\Scripts\python.exe -m src.training.train --faces-index data/processed/faces.csv --epochs 10 --batch-size 16 --learning-rate 0.0003
```

This writes `checkpoints/best_model.pt`, `checkpoints/final_model.pt`, `results/training/history.csv`, and run metadata. It prints device, model, total parameters, and trainable parameters.

### Evaluate only the held-out test split

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.evaluate --checkpoint checkpoints/best_model.pt --faces-index data/processed/faces.csv --results-dir results
```

This creates test predictions, metric JSON, a confusion matrix, ROC/precision-recall curves where both test classes are present, and actual training curves from checkpoint history.

### Image inference

```powershell
.\.venv\Scripts\python.exe -m src.inference.image_predict --input path\to\image.jpg --checkpoint checkpoints/best_model.pt
```

If no face is detected, the result is `UNKNOWN`; it does not force a label.

### Video inference

```powershell
.\.venv\Scripts\python.exe -m src.inference.video_predict --input path\to\video.mp4 --checkpoint checkpoints/best_model.pt --every-n-frames 15 --max-frames 64
```

For each valid face frame, the model produces raw softmax `P(FAKE)`. The final video score is:

```text
mean_fake_probability = sum(P(FAKE) for valid face frames) / valid_face_frames
```

The final class uses a 0.5 threshold. Confidence is `max(P(REAL), P(FAKE))`; it is not a calibrated confidence estimate. If there are no valid face frames, the result is `UNKNOWN`. The command writes a frame-level CSV and JSON forensic report.

### Unit tests and coverage

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest --cov=src --cov-report=term-missing --cov-report=html
```

## Review-2 evidence status

| Evidence | Status |
| --- | --- |
| Dataset split report | NOT YET RUN — local dataset not supplied |
| Training checkpoint | NOT YET RUN — local dataset not supplied |
| Held-out test metrics and plots | NOT YET RUN — no trained checkpoint or held-out data |
| Image/video forensic report | NOT YET RUN — no trained checkpoint supplied |
| Pytest | VERIFIED: 19 passed (2026-09-21) |
| Coverage report | VERIFIED: 49% line/branch coverage; `htmlcov/index.html` generated (2026-09-21) |

## Limitations and future work

This is a baseline detector, not proof of authenticity and not a manipulation-method classifier. Haar detection can miss non-frontal, obscured, or low-resolution faces. Raw probability scores are not calibrated. Future work includes stronger face detectors, identity-aware splitting where dataset identity metadata is available, score calibration, cross-dataset testing, and temporal models.
