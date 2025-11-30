# Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection

A production-ready, dual-task classification system using fine-tuned RoBERTa models for comprehensive LLM safety analysis:
1. **Refusal Classification** (3 classes): No Refusal, Hard Refusal, Soft Refusal
2. **Jailbreak Detection** (2 classes): Attack Failed, Attack Succeeded

---

## Overview

This project implements a complete pipeline for training, evaluating, and deploying dual RoBERTa classifiers that analyze LLM responses for safety-critical patterns. The system collects responses from multiple LLMs (Claude, GPT, Gemini), labels them using an LLM judge, and trains specialized classifiers with integrated cross-validation.

### Key Features

- **Dual-Task Classification**: Simultaneously detects refusal patterns AND jailbreak success
- **Integrated Cross-Validation**: K-fold CV built into training pipeline with statistical reporting
- **Multi-Model Data Collection**: Collects responses from Claude Sonnet 4.5, GPT-5, Gemini 2.5 Flash
- **LLM Judge Labeling**: GPT-4o-based labeling with confidence scores
- **WildJailbreak Integration**: Automatic data supplementation from HuggingFace datasets
- **Parallel Processing**: ThreadPoolExecutor-based concurrent API calls (5-10x speedup)
- **Checkpoint Recovery**: Automatic resume from crashes/interruptions
- **PDF Report Generation**: Professional reports using ReportLab
- **Production API**: FastAPI server with health checks and monitoring
- **Docker Support**: Multi-stage builds with GPU support

---

## Why RoBERTa?

**RoBERTa** (Robustly Optimized BERT Pretraining Approach) was selected based on literature demonstrating its superiority for text classification, particularly in safety-critical domains:

1. **Superior Text Classification Performance** (Liu et al., 2019)
   - State-of-the-art results on GLUE, RACE, and SQuAD benchmarks
   - Outperforms BERT through dynamic masking, larger batch sizes, removal of NSP

2. **Robust for Safety & Toxicity Detection** (Vidgen et al., 2021; Pozzobon et al., 2023)
   - Consistently outperforms alternatives on hate speech and toxic content detection

3. **Effective for Refusal Pattern Recognition** (Qi et al., 2023)
   - Bidirectional attention mechanism critical for understanding subtle refusal cues

4. **Production-Ready & Well-Supported**
   - Extensive Hugging Face ecosystem with efficient fine-tuning

**Key References:**
- Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach." *arXiv:1907.11692*
- Vidgen, B., et al. (2021). "Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection." *ACL 2021*

---

## Project Structure

```
Dual-RoBERTa-Classifiers/
├── src/
│   ├── 01-Imports.py               # Central import manager
│   ├── 02-Setup.py                 # Environment, paths, device config, class labels
│   ├── 03-Utils.py                 # Utility functions (KeepAwake, DynamicRateLimiter)
│   ├── 04-Config.py                # All configuration settings (Control Panel)
│   ├── 05-CheckpointManager.py     # Checkpoint management for error recovery
│   ├── 06-AWS.py                   # AWS configuration and Secrets Manager
│   ├── 07-PromptGenerator.py       # 3-stage prompt generation with personas
│   ├── 08-ResponseCollector.py     # Multi-LLM response collection + parallel processing
│   ├── 09-DataCleaner.py           # Comprehensive data cleaning
│   ├── 10-DataLabeler.py           # LLM judge labeling + parallel processing
│   ├── 11-WildJailbreakLoader.py   # WildJailbreak dataset loader for supplementation
│   ├── 12-LabelingQualityAnalyzer.py # Quality analysis for labeled data
│   ├── 13-ClassificationDataset.py # PyTorch Dataset
│   ├── 14-DatasetValidator.py      # Statistical hypothesis testing for datasets
│   ├── 15-RefusalClassifier.py     # 3-class RoBERTa model
│   ├── 16-JailbreakDetector.py     # 2-class RoBERTa model
│   ├── 17-Trainer.py               # Trainer with weighted loss + mixed precision
│   ├── 18-CrossValidator.py        # K-fold cross-validation with statistics
│   ├── 19-PerModelAnalyzer.py      # Per-model performance analysis
│   ├── 20-ConfidenceAnalyzer.py    # Confidence score analysis
│   ├── 21-AdversarialTester.py     # Paraphrasing robustness tests
│   ├── 22-JailbreakAnalysis.py     # Security-focused jailbreak analysis
│   ├── 23-CorrelationAnalysis.py   # Refusal <-> Jailbreak correlation
│   ├── 24-AttentionVisualizer.py   # Attention heatmaps
│   ├── 25-ShapAnalyzer.py          # SHAP interpretability
│   ├── 26-PowerLawAnalyzer.py      # Power law analysis
│   ├── 27-ErrorAnalysis.py         # Comprehensive error analysis
│   ├── 28-Visualizer.py            # Basic plotting functions
│   ├── 29-ReportGenerator.py       # PDF report generation
│   ├── 30-RefusalPipeline.py       # Main training pipeline
│   ├── 31-ExperimentRunner.py      # Experiment orchestration
│   ├── 32-Execute.py               # Main entry point
│   ├── 33-Analyze.py               # Standalone analysis script
│   ├── 34-ProductionAPI.py         # FastAPI server
│   ├── 35-MonitoringSystem.py      # Production monitoring
│   ├── 36-RetrainingPipeline.py    # Automated retraining
│   └── 37-DataManager.py           # Production data management
├── Data/                           # Dataset files (CSV, JSON, pickle)
├── Reports/                        # Generated PDF reports
├── data/                           # Runtime data (created automatically)
├── models/                         # Trained models (created automatically)
├── results/                        # Analysis results (created automatically)
├── visualizations/                 # Generated plots (created automatically)
├── requirements.txt                # Core Python dependencies
├── requirements-aws.txt            # AWS-specific dependencies
├── requirements-interpretability.txt # SHAP dependencies
├── Dockerfile                      # Docker multi-stage build
├── docker-compose.yml              # Docker Compose configuration
├── .env.template                   # Environment variables template
├── LICENSE                         # MIT License
└── README.md                       # This file
```

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/ramyalsaffar/Dual-RoBERTa-Classifiers-for-3-Class-Refusal-Taxonomy-and-Binary-Jailbreak-Detection.git
cd Dual-RoBERTa-Classifiers-for-3-Class-Refusal-Taxonomy-and-Binary-Jailbreak-Detection

# Install dependencies
pip install -r requirements.txt

# Optional: Install SHAP for interpretability
pip install -r requirements-interpretability.txt
```

### 2. Set API Keys

```bash
# Required API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

Or copy `.env.template` to `.env` and fill in your keys.

### 3. Run Experiments

```bash
# Interactive mode (recommended for first run)
python src/32-Execute.py

# Command-line modes
python src/32-Execute.py --test           # Quick test with reduced samples
python src/32-Execute.py --full           # Full experiment with cross-validation
python src/32-Execute.py --analyze-only   # Analyze existing models
```

### 4. Generate Reports

```bash
# Auto-detect models and generate all reports
python src/33-Analyze.py --auto --generate-report

# Specify report type
python src/33-Analyze.py --auto --generate-report --report-type performance
python src/33-Analyze.py --auto --generate-report --report-type executive
```

---

## Docker Deployment

### Quick Start with Docker

```bash
# Build the image
docker-compose build dev

# Start development environment
docker-compose up dev

# Or use specific services
docker-compose up train      # Full training pipeline
docker-compose up analyze    # Analysis only
docker-compose up api        # Production API server
docker-compose up jupyter    # Jupyter notebook
```

### Environment Setup

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
```

### Docker Services

| Service | Purpose | Command |
|---------|---------|---------|
| `dev` | Interactive development | `docker-compose up dev` |
| `train` | Full training pipeline | `docker-compose up train` |
| `analyze` | Analysis with reports | `docker-compose up analyze` |
| `api` | Production API server | `docker-compose up api` |
| `jupyter` | Jupyter notebook | `docker-compose up jupyter` |

---

## Pipeline Architecture

### Data Pipeline
```
Prompt Generation (3-stage) -> Multi-LLM Response Collection -> Data Cleaning -> LLM Judge Labeling -> Quality Analysis -> Train/Val/Test Split
```

### Training Pipeline
```
Load Data -> Create Dataset -> Initialize Model (RoBERTa) -> Weighted Loss -> K-Fold Cross-Validation -> Train with Early Stopping -> Save Best Model
```

### Analysis Pipeline
```
Load Model -> Per-Model Analysis -> Confidence Analysis -> Error Analysis -> Visualizations -> PDF Reports
```

---

## Model Performance

**Refusal Classifier (3-class):**
- Classes: No Refusal (0), Hard Refusal (1), Soft Refusal (2)
- Model: RoBERTa-base fine-tuned
- Training: Weighted CrossEntropyLoss (handles class imbalance)
- Evaluation: F1-score (macro), Precision, Recall, Accuracy

**Jailbreak Detector (2-class):**
- Classes: Attack Failed (0), Attack Succeeded (1)
- Model: RoBERTa-base fine-tuned
- Focus: Security-critical detection with WildJailbreak supplementation

---

## Configuration

All settings are centralized in `src/04-Config.py`:

- **API_CONFIG**: Models, rate limits, retries
- **DATASET_CONFIG**: Sample sizes, splits, random seed
- **TRAINING_CONFIG**: Epochs, batch size, learning rate, early stopping
- **MODEL_CONFIG**: Architecture, num_classes, dropout
- **CROSS_VALIDATION_CONFIG**: K-folds, metrics
- **WILDJAILBREAK_CONFIG**: Dataset supplementation settings

---

## Interpretability

The project includes comprehensive interpretability tools:

1. **SHAP Analysis**: Token-level feature importance
2. **Attention Visualization**: Multi-head attention heatmaps
3. **Confidence Analysis**: Calibration and uncertainty
4. **Error Analysis**: Systematic failure case examination
5. **Power Law Analysis**: Pareto principle in predictions

---

## Security

**Important:** Never commit API keys or sensitive data!

- All API keys via environment variables
- AWS Secrets Manager integration available
- `.gitignore` configured to exclude secrets
- `.env` files excluded from version control

---

## Dependencies

See `requirements.txt` for full list. Key dependencies:

- **PyTorch** (2.0+): Deep learning framework
- **Transformers** (4.30+): Hugging Face models
- **FastAPI** (0.100+): Production API
- **ReportLab** (4.0+): PDF generation
- **SHAP** (0.42+): Interpretability (optional)

---

## License

MIT License - see LICENSE file for details.

---

## Author

Ramy Alsaffar

---

## Acknowledgments

### Models & Libraries
- **RoBERTa**: Liu et al., 2019 - [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- **SHAP**: Lundberg & Lee, 2017 - [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- **Transformers**: Hugging Face team - [State-of-the-art NLP](https://github.com/huggingface/transformers)

### Datasets
- **WildJailbreak**: AllenAI, 2024 - Used for supplementing jailbreak detection training data
  - Paper: [WildTeaming at Scale](https://arxiv.org/abs/2406.18510)
  - Dataset: [allenai/wildjailbreak on Hugging Face](https://huggingface.co/datasets/allenai/wildjailbreak)
  - License: Apache 2.0

---

**Last Updated:** November 30, 2025
