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

---

# HOW TO RUN

To download the dataset, follow these steps after cloning the repository:

1. Make sure you have a **Kaggle API token** on your computer.  
   Follow the official Kaggle guide to create and configure your token: [Kaggle API Guide](https://www.kaggle.com/docs/api).

2. Run the dataset script to download the data:

```bash
python src/credit_card_fraud_analysis/make_dataset.py
```

3. Run the training script:

```bash
python src/credit_card_fraud_analysis/train.py
```

4. Run the evaluate script:

```bash
python src/credit_card_fraud_analysis/evaluate.py
```