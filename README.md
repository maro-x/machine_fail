# Machine Failure Prediction

## Overview
This project predicts machine failures based on sensor readings and operational metrics. It uses a **feed-forward neural network (PyTorch)** to classify whether a machine will fail (`fail=1`) or not (`fail=0`) based on input features.

The project also provides a **FastAPI** interface for deploying the model as an API.

---

## Dataset
- Dataset contains sensor readings (`footfall`, `AQ`, `USS`, `CS`, `VOC`, `RP`, `IP`, `Temperature`) and a categorical feature (`tempMode`).
- Target column is `fail` (1 for failure, 0 for normal).
- The dataset is preprocessed:
  - Numerical features are **standardized** using `StandardScaler`.
  - Categorical features are kept as is.
  - Preprocessed data is saved as `preprocessed.csv`.

---

## Requirements
Python 3.8+ with the following packages:
- pandas
- numpy
- scikit-learn
- torch
- fastapi
- pydantic
- uvicorn
- pickle

Install via:

```bash
pip install pandas numpy scikit-learn torch fastapi pydantic uvicorn

machine_failure/
│
├─ data/
│   ├─ data.csv               # Raw dataset
│   ├─ preprocessed.csv       # Preprocessed dataset
│   └─ scaler.pkl             # StandardScaler object
│
├─ model/
│   └─ model.pt               # Trained PyTorch model
│
├─ infere.py                  # Model inference functions
├─ main.py                    # FastAPI app
├─ train.py                   # Training script
└─ requirements.txt
