# Credit Card Fraud Detection Project

## 1. Overall goal

Main goal of this project is to design and evaluate a machine learning system capable of detecting fraudulent credit card transactions efficiently. Credit card fraud is a critical real-world problem characterized by highly imbalanced data and evolving fraud patterns. By accurately distinguishing between legitimate and malicious activity, the project aims to minimize financial losses for bank institutions and protect consumers from unauthorized charges.

---

## 2. Frameworks and Tools

The project leverages the following technologies and tools:

- **Python**
- **Uv** - Create virtual environments and manage dependencies
- **Git** – Version control
- **scikit-learn** – Machine learning algorithms and preprocessing pipelines
- **PyTorch** – Neural network implementation
- **Pandas & NumPy** – Data handling and cleaning
- **Matplotlib & Seaborn** – Data visualization
- **Cookiecutter** – Standardized project structure and reproducibility
- **Hydra & YAML** – Dynamic configuration and hyperparameter management
- **Weights & Biases (W&B)** – Experiment tracking and performance logging
- **Docker** – Containerization and environment portability
- **Pytorch Lightning** - Reduce boilerplate code

---

## 3. Data Sources

The Kaggle Credit Card Fraud Detection dataset containing 284,807 transactions made by European cardholders in September 2013. Features V1-V28 are PCA-transformed, plus Time and Amount. This dataset presents a significant challenge due to its extreme class imbalance, where only 0.17\% of the 284,807 transactions are labeled as fraudulent

---

## 4. Models

A variety of models can be deployed but initially an Autoencoder that learns to reconstruct normal (non-fraudulent) transactions. An Autoencoder is a neural network used for unsupervised anomaly detection by learning to compress and reconstruct data. In the context of credit card fraud, the model is trained exclusively on "normal" transactions to learn the standard patterns of legitimate behavior. When the model encounters a fraudulent transaction, it lacks the specialized knowledge to reconstruct it accurately, resulting in a significantly high reconstruction error, which serves as a clear signal to flag the transaction as suspicious. This approach is particularly valuable because it does not rely on a large set of labeled fraud examples, which are often rare in real-world datasets.

## 5. Training Pipeline and Experiment Tracking

Model training is implemented using **PyTorch Lightning** to reduce boilerplate code abd enforec a clean seperation between model definition, optimization, and training logic. The autoencoder is implemented as a LightingModule, while training is manahged through the Lightning Trainer abstraction, enabling standardized logging,  checkpointing, and hardware-agnostic execution.

Experiment tracking is handled using **Weighs & Biases (W&B)**. During training, step-level reconstruction losses as well as optimizer learning rates are logged automatically. Model checkpoints are saved usign a ModelCheckpoont callback, and the best-performing model is stored locally and tracked as an artifact. Each training run is versioned and linked to its coresponding metrics and cofniguration in the W&B dashboard. Basic runtime profiling can be enabled via configuration using PyTorch Lightning's built-in profiler to identify potential training and data-loading bottlenecks.

---

# HOW TO RUN


1. Setup environmet. Install dependencies:

```bash
pip install -r requirements.txt```

To download the dataset, follow these steps after cloning the repository:

2. Make sure you have a **Kaggle API token** on your computer.
   Follow the official Kaggle guide to create and configure your token: [Kaggle API Guide](https://www.kaggle.com/docs/api).


3. Run the dataset script to download the data:

```bash
PYTHONPATH=src python src/credit_card_fraud_analysis/make_dataset.py
```

4. If you want training runs to be logged to the wandb dahsboard:

`wandb login`
 

3. Run the training script using PyTorch and wandb:

```bash
PYTHONPATH=src python src/credit_card_fraud_analysis/train_lightning.py
```

4. Run the evaluate script:

```bash
PYTHONPATH=src python src/credit_card_fraud_analysis/evaluate.py
```
