# Specificity Bias and the Wikipedia Paradox: An xAI Analysis of AI Text Detectors under Domain Shift

[![arXiv](https://img.shields.io/badge/arXiv-Preprint-B31B1B.svg)](https://arxiv.org/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official code, datasets, and Explainable AI (xAI) visualizations for the master's project: **"Specificity Bias and the Wikipedia Paradox: An xAI Analysis of AI Text Detectors under Domain Shift"** by Nico Morloher.

## 📌 Overview
Current Transformer-based AI text detectors (like RoBERTa and DistilBERT) often experience severe performance degradation during domain shifts and operate as opaque black-box classifiers. This project investigates their robustness and interpretability by evaluating them against a custom out-of-distribution (OOD) dataset of hyper-local news synthesized via Llama 3, Mistral, and Gemma. 

Using a tripartite **SHAP (SHapley Additive exPlanations)** framework, we expose the exact decision boundaries of these models at the token level, revealing three key phenomena:
1. **The "Call-to-Action" Syndrome:** Detectors heavily rely on the assistance-oriented tone inherent to instruction-tuned LLMs rather than generic machine text.
2. **The "Specificity Bias":** LLMs successfully bypass detection (False Negatives / "Sneaky AI") by hallucinating specific Named Entities and mimicking conversational imperfection.
3. **The "Wikipedia Paradox":** Formal, objective human writing is frequently misclassified as AI (False Positives) due to its low statistical burstiness and factual density.

## 📂 Repository Structure

```text
├── data/
│   ├── raw/                 # Original datasets (HC3 PLUS, artem9k, GPT-wiki-intro)
│   └── processed/           # Custom hyper-local OOD dataset (News)
├── models/                  # Directory for fine-tuned checkpoints (DistilBERT, RoBERTa) (has to be created on your machine!)
├── src/
│   ├── 1_data_generation/   # Scripts for dataset curation and LLM generation (Ollama)
│   ├── 2_training/          # Fine-tuning scripts for the Transformer models
│   └── 3_evaluation/        # Model evaluation and SHAP token-level attribution scripts
├── results/                 # High-resolution SHAP HTML heatmaps, AUROC/AUPRC curves
├── README.md
├── requirements.txt
├── project_structure.md            
└── FINAL_local_news_validation.csv
```

## 🚀 Installation & Setup
1. Clone the repository:
```bash
   git clone [https://github.com/morloher779/xai-analysis.git](https://github.com/morloher779/xai-analysis.git)
   cd xai-analysis
```
2. Create a virtual environment and install dependencies:
```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip install -r requirements.txt
```
(Note: Ensure PyTorch is installed with the correct CUDA version for your local hardware).

## 🧠 Usage
# 1. Training the Models
To fine-tune the Transformer architectures (DistilBERT or RoBERTa) on the respective datasets, run:
```bash
  python src/2_training/train_roberta_megacorpus.py
```
# 2. Evaluating Performance
To generate the quantitative metrics (Accuracy, F1, AUROC, AUPRC) across the OOD domains:
```bash
  python src/3_evaluation/evaluate_roberta_megacorpus.py
```
# 3. Generating xAI Visualizations (SHAP)
To recreate the interactive token-level SHAP heatmaps presented in the paper (e.g., comparing 6-layer DistilBERT vs. 12-layer RoBERTa):
```bash
  python src/3_evaluation/generate_shap_true_positives.py
  python src/3_evaluation/generate_shap_false_negatives.py
```
The resulting HTML files will be saved in the results/ directory.

## 📝 Citation
If you find this code, the dataset, or the xAI heatmaps useful for your research, please consider citing the corresponding paper:
```latex
@unpublished{morloher2026specificity,
  title={Specificity Bias and the Wikipedia Paradox: An xAI Analysis of AI Text Detectors under Domain Shift},
  author={Morloher, Nico and Reschke, Johannes},
  note={Master's Project, Ostbayerische Technische Hochschule Regensburg (OTH)},
  year={2026}
}
```
## 🤝 Acknowledgements
This project utilizes the Hugging Face transformers library and the shap interpretability framework. Synthetic datasets were generated using open-source models via the Ollama framework.
