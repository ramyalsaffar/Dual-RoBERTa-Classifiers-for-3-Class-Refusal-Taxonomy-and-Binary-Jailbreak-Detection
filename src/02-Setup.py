# Setup
#------
# Device configuration, display settings, and project constants.
# True constants and initialization settings only.
###############################################################################


#------------------------------------------------------------------------------


# Device Configuration
#---------------------
# Auto-detect best available device
# =============================================================================
if torch.cuda.is_available():
     DEVICE = torch.device('cuda')
     try:
         print(f"🚀 CUDA available - using GPU: {torch.cuda.get_device_name(0)}")
     except:
         print("🚀 CUDA available - using GPU")
elif IS_MAC and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
     DEVICE = torch.device('mps')
     print("🚀 MPS available - using Apple Silicon GPU")
else:
     DEVICE = torch.device('cpu')
     print("⚠️  No GPU available - using CPU (training will be slow)")
 

# =============================================================================
# # Device Configuration
# # Check if this is a training run (Steps 6-7)
# IS_TRAINING_RUN = '--test' in sys.argv or '--full' in sys.argv
# 
# if IS_TRAINING_RUN:
#     # Force CPU for training stability (MPS memory leak with CV)
#     DEVICE = torch.device('cpu')
#     print("🔧 FORCED CPU MODE for training stability")
#     print("   Expected time: ~45 minutes (vs. MPS hangs after 9+ hours)")
# else:
#     # For non-training tasks, use GPU acceleration
#     if torch.cuda.is_available():
#         DEVICE = torch.device('cuda')
#     elif IS_MAC and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#         DEVICE = torch.device('mps')
#     else:
#         DEVICE = torch.device('cpu')
# 
# =============================================================================


#------------------------------------------------------------------------------


# Python Display Settings
#------------------------
pd.set_option('display.max_rows', 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_colwidth", 250)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 5)
pd.options.display.float_format = '{:.4f}'.format


#------------------------------------------------------------------------------


# =============================================================================
# CLASS LABELS AND MAPPINGS
# =============================================================================

# Refusal Classifier - Class names for display
#----------------------------------------------
CLASS_NAMES = ['No Refusal', 'Hard Refusal', 'Soft Refusal']

# Jailbreak Detector - Class names for display
#----------------------------------------------
JAILBREAK_CLASS_NAMES = ['Jailbreak Failed', 'Jailbreak Succeeded']


# =============================================================================
# COLUMN NAMING CONVENTIONS (IMPORTANT!)
# =============================================================================
#
# Three different jailbreak-related columns exist with distinct purposes:
#
# 1. 'is_jailbreak_attempt' (binary flag)
#    - Purpose: Identifies if a prompt is attempting a jailbreak
#    - Values: 0 = not a jailbreak attempt, 1 = is a jailbreak attempt
#    - Used by: DataLabeler to flag prompts for jailbreak classification
#    - Column type: Boolean indicator
#    - When used: During labeling phase to identify which prompts need jailbreak analysis
#
# 2. 'jailbreak_label' (ground truth from judge or dataset)
#    - Purpose: Ground truth label indicating whether jailbreak succeeded
#    - Values: 0 = jailbreak failed, 1 = jailbreak succeeded, -1 = error
#    - Used by: DataLabeler outputs, training datasets, analysis files
#    - Source: GPT-4 judge labeling OR dataset ground truth
#    - Column type: Ground truth label
#    - When used: Throughout pipeline for training, evaluation, and analysis
#    - Note: This is the PRIMARY column name used throughout the codebase
#
# 3. 'jailbreak_success' (ONLY in GPT-4 JSON response)
#    - Purpose: External API format from GPT-4 judge
#    - Values: 0 = jailbreak failed, 1 = jailbreak succeeded, -1 = error
#    - Used by: GPT-4 API JSON response format only
#    - When used: Internally in DataLabeler when parsing GPT-4 responses
#    - Note: Immediately mapped to 'jailbreak_label' in Python code (line 527 in DataLabeler)
#    - This name exists ONLY in the JSON - all Python code uses 'jailbreak_label'
#
# Why this naming?
# - 'is_jailbreak_attempt' = "Is this a jailbreak attempt?" (filtering)
# - 'jailbreak_label' = "Did the jailbreak succeed?" (ground truth)
# - 'jailbreak_success' = External JSON field name only (not used in Python variables)
#
# IMPORTANT: After DataLabeler processes responses, everything uses 'jailbreak_label'.
# The 'jailbreak_success' name only appears in:
#   1. GPT-4's JSON response format
#   2. Old checkpoint files (handled by backward compatibility in RefusalPipeline line 469-470)
#


# =============================================================================
# VISUALIZATION COLORS
# =============================================================================

# Refusal Class colors (for plots)
#-----------------------------------
PLOT_COLORS = {
    'no_refusal': '#2ecc71',        # Green
    'hard_refusal': '#e74c3c',      # Red
    'soft_refusal': '#f39c12'       # Orange
}

# Class colors as list (for matplotlib) - derived from dict
PLOT_COLORS_LIST = list(PLOT_COLORS.values())

# Jailbreak Class colors (for plots)
#------------------------------------
JAILBREAK_PLOT_COLORS = {
    'jailbreak_failed': '#27ae60',          # Dark Green (success - model defended)
    'jailbreak_succeeded': '#c0392b'        # Dark Red (failure - model broken)
}

# Jailbreak colors as list (for matplotlib) - derived from dict
JAILBREAK_PLOT_COLORS_LIST = list(JAILBREAK_PLOT_COLORS.values())

# Model colors (for per-model analysis)
#---------------------------------------
# Keys match API_CONFIG['response_models'] keys in 03-Config.py
MODEL_COLORS = {
    'claude': '#3498db',         # Blue - Claude Sonnet
    'gpt5': '#9b59b6',          # Purple - GPT-5
    'gemini': '#e67e22'         # Orange - Gemini 2.5 Flash
}


# =============================================================================
# ERROR HANDLING CONSTANTS
# =============================================================================

# Error indicator value (used to mark failed evaluations)
ERROR_VALUE = -1

# Error response placeholder
ERROR_RESPONSE = "[ERROR]"


# =============================================================================
# WILDJAILBREAK DATASET INFORMATION
# =============================================================================

# WildJailbreak Dataset - Used for supplementing jailbreak training data
#------------------------------------------------------------------------
# When modern LLMs successfully defend against all jailbreak attempts,
# we supplement training data from the WildJailbreak dataset to ensure
# the jailbreak detector has sufficient positive samples.

WILDJAILBREAK_DATASET_INFO = {
    'name': 'WildJailbreak',
    'source': 'AllenAI',
    'url': 'https://huggingface.co/datasets/allenai/wildjailbreak',
    'paper_url': 'https://arxiv.org/abs/2406.18510',
    'paper_title': 'WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models',
    'authors': 'Liwei Jiang, Kavel Rao, Seungju Han, et al.',
    'conference': 'NeurIPS',
    'year': '2025',
    'size': '262K prompt-response pairs',
    'adversarial_harmful_samples': '82,728',  # Successful jailbreaks
    'license': 'Apache 2.0'
}

WILDJAILBREAK_CITATION = """
@inproceedings{jiang2025wildteaming,
    title={WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models},
    author={Liwei Jiang and Kavel Rao and Seungju Han and Allyson Ettinger and Faeze Brahman and Sachin Kumar and Niloofar Mireshghallah and Ximing Lu and Maarten Sap and Yejin Choi and Nouha Dziri},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2025},
    volume={38},
    url={https://arxiv.org/abs/2406.18510}
}
"""

# Dataset Acknowledgment
WILDJAILBREAK_ACKNOWLEDGMENT = """
This project uses the WildJailbreak dataset from AllenAI for supplementing
jailbreak detection training data when insufficient positive samples are
collected from our primary pipeline. WildJailbreak provides diverse,
in-the-wild jailbreak tactics that enhance model robustness.
"""


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
