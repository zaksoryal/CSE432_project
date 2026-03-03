# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Speech Emotion Recognition (SER) semester project for CSE 432/532 at the University of Miami. The project uses the RAVDESS audio dataset (~2,452 WAV files) and requires building a from-scratch ML library called **MiniLearn**, comparing it against scikit-learn, and evaluating multiple supervised/unsupervised models.

## Environment Setup

```bash
# Create and activate virtual environment from repo root
python -m venv .venv
source .venv/bin/activate           # macOS/Linux
.venv\Scripts\activate              # Windows (Command Prompt)

# Install dependencies
pip install -r SER_Project/requirements.txt

# Download RAVDESS dataset (saves to data/ in SER_Project/)
cd SER_Project
python download_data.py
```

## Project Structure

```
Project_CSE432-532/
├── README.md                   # Full project specification and grading rubric
├── data/                       # RAVDESS audio files, Actor_01/ to Actor_24/
├── Papers/                     # Reference research papers
└── SER_Project/
    ├── requirements.txt
    ├── download_data.py
    └── minilearn/              # From-scratch ML library (main deliverable)
        └── __init__.py
```

## Architecture

### MiniLearn Library (`SER_Project/minilearn/`)
Must be implemented from scratch using only NumPy/SciPy — no scikit-learn internally. Required components:
- **Classifiers:** Logistic Regression (gradient descent), KNN, Gaussian Naive Bayes, Decision Tree (CART), SVM (linear), ANN
- **Preprocessing:** `StandardScaler`, `train_test_split`
- **Metrics:** Accuracy, Precision, Recall, F1, Confusion Matrix, k-fold cross-validation
- **Dimensionality Reduction:** PCA
- **Clustering:** K-Means

### Data Pipeline
- Audio files follow naming convention `03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav`
  - Position 3 (1-indexed) = emotion class (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
- Feature extraction uses `librosa`: MFCCs (13 coeff), MFCC deltas, chroma, mel spectrogram, ZCR, RMS, spectral centroid/bandwidth/rolloff
- Per-frame features → compute mean/std/min/max → fixed-length feature vector per file

### Critical ML Rules
- **Always** fit `StandardScaler` on training data only; transform test data with the fitted scaler
- Use Stratified K-Fold Cross-Validation (k=5 or 10) for all model evaluations
- Every MiniLearn model must be compared against the scikit-learn equivalent

## Deliverables (Jupyter Notebooks)
The final report is notebook-based. Planned structure:
- EDA and feature extraction notebook
- MiniLearn implementation and validation notebook
- Supervised classification comparison notebook
- Unsupervised clustering and PCA/t-SNE notebook
- Neural network models notebook (DNN, 1D-CNN, LSTM/GRU)

## Grading Rubric Summary (100 pts)
| Section | Pts | Key requirement |
|---------|-----|-----------------|
| Data & EDA | 10 | Filename parsing, class distribution, visualizations |
| Feature Extraction | 10 | Multiple feature types, proper standardization |
| MiniLearn Library | 30 | All classifiers + preprocessing + metrics from scratch |
| Supervised Classification | 20 | Classical + ≥1 NN; MiniLearn vs sklearn comparison |
| Evaluation & Validation | 10 | All metrics, CV, hyperparameter tuning |
| Unsupervised / Clustering | 10 | K-Means + Hierarchical, ARI/NMI, PCA/t-SNE |
| Report Quality | 10 | Critical analysis, not just plots/tables |
