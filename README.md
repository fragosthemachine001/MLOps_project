# Credit Card Fraud Detection Project

## 1. Overall goal

The goal of this project is to develop a machine learning system for credit card fraud detection within a reproducible MLOps framework. Fraud detection is a challenging real-world problem due to extreme class imbalance and evolving fraud patterns.

The task is formulated as an unsupervised anomaly detection problem, where an autoencoder is trained on normal transactions and fraudulent activity is identified using reconstruction error during inference. In addition to model development, the project emphasizes best practices in MLOps, including automated testing, continuous integration, containerization, and reproducibility across environments.

---
## Project Structure

The **Credit Card Fraud Detection Project** follows a modular, MLOps-oriented repository structure.
The complete directory layout is shown below.

<details>
<summary><strong>Click to expand full project structure</strong></summary>

```text
.
├── LICENSE
├── README.md
├── app.log
├── cloudbuild.train.yaml
├── cml_data.yaml
├── configs
│   ├── __init__.py
│   └── config.yaml
├── coverage.xml
├── data
│   ├── processed
│   └── raw
├── data.dvc
├── dockerfiles
│   ├── Dockerfile
│   ├── api.dockerfile
│   └── train.dockerfile
├── docs
│   ├── README.md
│   ├── mkdocs.yaml
│   ├── profiling.md
│   └── source
├── front
│   ├── frontend.py
│   ├── frontend.dockerfile
│   ├── run_api.bat
│   ├── run_frontend.bat
│   ├── run_training.bat
│   └── start_system.bat
├── logs
│   └── app.log
├── models
│   ├── autoencoder.pt
│   └── lit-autoencoder-epoch=00-train_loss=0.3852.ckpt
├── notebooks
├── prediction_database.csv
├── pyproject.toml
├── pytest.ini
├── reports
│   ├── README.md
│   ├── figures
│   └── report.py
├── requirements.txt
├── requirements_dev.txt
├── src
│   └── credit_card_fraud_analysis
├── tasks.py
├── tests
│   ├── integration_tests
│   ├── performance_tests
│   └── unit tests
├── uv.lock
└── wandb

## 2. Frameworks and Tools

The project leverages the following technologies and tools:

- **Python**
- **Uv**, "Pip" - Create virtual environments and manage dependencies
- **Git** – Version control
- **PyTorch** – Neural network implementation
- **Pandas & NumPy** – Data handling and cleaning
- **Cookiecutter** – Standardized project structure and reproducibility
- **Hydra & YAML** – Dynamic configuration and hyperparameter management
- **Weights & Biases (W&B)** – Experiment tracking and performance logging
- **Docker** – Containerization and environment portability
- **Pytorch Lightning** - Reduce boilerplate code
- **Typer** - used to expose parts of the pipeline (e.g. data preparation, training, and evaluation)
- **DVC** - Data management and reproducibility
- **GitHub Actions** - Continuous integration
- **pytest** - Code quality
- **Ruff** - linting and formatting
- **Google Cloud Platform (GCP)** using a Google Cloud Storage bucket for remote storage.

---

## 3. Data Sources

The Kaggle Credit Card Fraud Detection dataset containing 284,807 transactions made by European cardholders in September 2013. Features V1-V28 are PCA-transformed, plus Time and Amount. This dataset presents a significant challenge due to its extreme class imbalance, where only 0.17\% of the 284,807 transactions are labeled as fraudulent. You can find the datasource here:
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

---

## 4. Models

The project uses an **autoencoder neural network** for fraud detection via **unsupervised anomaly detection**. The model is trained exclusively on legitimate (non-fraudulent) credit card transactions to learn normal behavioral patterns.

At inference time, transactions that cannot be accurately reconstructed by the autoencoder produce a **high reconstruction error**, which is used as an anomaly score to flag potentially fraudulent activity. This approach is well suited to credit card fraud detection, where labeled fraud examples are scarce and the dataset is highly imbalanced.

The model is implemented in **PyTorch** and trained using **PyTorch Lightning** to ensure a clean and reproducible training pipeline.

## 5. Training Pipeline and Experiment Tracking

Model training is implemented using **PyTorch Lightning**, which provides a structured training loop and enforces a clear separation between model definition, optimization, and training logic. Training is executed via the Lightning `Trainer`, enabling standardized logging, checkpointing, and hardware-agnostic execution.

Experiment tracking is handled using **Weights & Biases (W&B)**. During training, reconstruction loss and optimizer metrics are logged automatically. Model checkpoints are saved using a `ModelCheckpoint` callback, and each training run is versioned and linked to its configuration and metrics in the W&B dashboard, supporting reproducibility and experiment comparison.

---

# HOW TO RUN


1. Create a virtual environment. Install dependencies:

```bash
pip install -r requirements.txt
```

2. To download the dataset, follow these steps after cloning the repository:

- Make sure you have a **Kaggle API token** on your computer.
   Follow the official Kaggle guide to create and configure your token: [Kaggle API Guide](https://www.kaggle.com/docs/api).


- Run the dataset script to download the data:

```bash
PYTHONPATH=src python src/credit_card_fraud_analysis/make_dataset.py
$env:PYTHONPATH="src"; python src/credit_card_fraud_analysis/make_dataset.py
```

3. If you want training runs to be logged to the wandb dahsboard:

`wandb login`


4. Run the training script using PyTorch and wandb:

```bash
PYTHONPATH=src python src/credit_card_fraud_analysis/train_lightning.py
$env:PYTHONPATH="src"; python src/credit_card_fraud_analysis/train_lightning.py
```

5. Run the evaluate script:

```bash
PYTHONPATH=src python src/credit_card_fraud_analysis/evaluate.py
$env:PYTHONPATH="src"; python python src/credit_card_fraud_analysis/evaluate.py
```

6. Run frontend:

```bash
front\start_system.bat
```
