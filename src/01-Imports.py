# Imports
#--------
#
# This file has most of the libraries needed for the project.
# Paths to load from or to.
#
###############################################################################


#------------------------------------------------------------------------------

# Standard library imports
import os
import sys
import json
import time
import re
import warnings
import subprocess
import signal
import pickle
import getpass
import atexit
import asyncio
import threading
import argparse
import traceback
import gc
#from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob

# ML Tools
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
    recall_score,
    precision_score,
    precision_recall_curve,
    average_precision_score,
    cohen_kappa_score,
    ConfusionMatrixDisplay,
    matthews_corrcoef
)


# Statistical Testing
from scipy.stats import chisquare, chi2_contingency, shapiro, kstest, chi2, binomtest
from scipy import stats as scipy_stats

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW

# Transformers
import transformers
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    RobertaConfig,
    get_linear_schedule_with_warmup
)


#warnings.filterwarnings('ignore')
#warnings.filterwarnings('ignore', category=FutureWarning)  # Only suppress specific warnings

# WildJailBreak dataset
import datasets
from datasets import load_dataset

from datasketch import MinHash, MinHashLSH

# Visualization
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import seaborn as sns
import shap

# PDF Report Generation (optional - only needed for report generation)
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, KeepTogether
    )
    from reportlab.platypus.flowables import HRFlowable
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("ℹ️  reportlab not available - PDF report generation disabled")

# API Clients
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import tiktoken  # Token counting for API usage tracking

# Environment variables
from dotenv import load_dotenv

# AWS (optional - only needed for cloud deployment)
try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    print("ℹ️  boto3 not available - AWS features disabled")


#------------------------------------------------------------------------------


# Environment Detection (Smart Defaults)
#----------------------------------------
# Detects if running locally or in AWS/Docker
# Defaults to 'local' if ENVIRONMENT variable not set
ENVIRONMENT = os.getenv('ENVIRONMENT', 'local')
IS_MAC = sys.platform == 'darwin'
IS_AWS = ENVIRONMENT == 'aws'

print(f"🌍 Running in: {ENVIRONMENT.upper()} mode")
if IS_MAC:
    print("🍎 Mac detected - using MPS acceleration if available")


#------------------------------------------------------------------------------


# Environment-Aware Paths
#------------------------
# Sets appropriate paths based on where code is running
if IS_AWS:
    # AWS/Docker paths
    project_path = "/app"
    base_results_path = "/app/results/"
    CodeFilePath = "/app/src/"
else:
    # Local paths (Mac default)
    main_path = "/Users/ramyalsaffar/Ramy/C.V..V/1-Resume/06- LLM Model Behavior Projects/"
    folder = "3-Class Refusal Classifier with RoBERTa"
    project_path = glob.glob(main_path + "*" + folder)[0]
    base_results_path = glob.glob(project_path + "/*Code/*Results")[0]


# Specific Subdirectories
#------------------------
# Data Subdirectories
data_path = glob.glob(base_results_path + "/*Data/")[0]
data_raw_path = glob.glob(data_path + "*Raw/")[0]
data_responses_path = glob.glob(data_path + "*Responses/")[0]
data_processed_path = glob.glob(data_path + "*Processed/")[0]
data_splits_path = glob.glob(data_path + "*Splits/")[0]
data_checkpoints_path = glob.glob(data_path + "*Checkpoints/")[0]

# Models Directory
models_path = glob.glob(base_results_path + "/*Models/")[0]

# Results Subdirectories
analysis_results_path = glob.glob(base_results_path + "/*Analysis")[0]
quality_review_path = glob.glob(base_results_path + "/*Quality Review")[0]

# Visualization Subdirectories
visualizations_path = glob.glob(base_results_path + "/*Visualizations/")[0]
attention_analysis_path = os.path.join(visualizations_path, "Attention Analysis")
shap_analysis_path = os.path.join(visualizations_path, "SHAP Analysis")
correlation_viz_path = os.path.join(visualizations_path, "Correlation Analysis")
error_analysis_path = os.path.join(visualizations_path, "Error Analysis")
power_law_viz_path = os.path.join(visualizations_path, "Power Law Analysis")
reporting_viz_path = os.path.join(visualizations_path, "For Reporting")

reporting_viz_path = os.path.join(visualizations_path, "For Reporting")

# Reports Directory
reports_path = glob.glob(base_results_path + "/*Reports/")[0]

# API Keys (local file storage)
api_keys_file_path = glob.glob(project_path + "/*API Keys/API Keys.txt")[0]

# Create subdirectories if they don't exist
for path in [analysis_results_path, quality_review_path, attention_analysis_path, 
             shap_analysis_path, correlation_viz_path, error_analysis_path, power_law_viz_path
             ,reporting_viz_path]:
    os.makedirs(path, exist_ok=True)


#------------------------------------------------------------------------------


# Execute Code Files
#-------------------
CodeFilePath = glob.glob(project_path + "/*Code/*Python/")[0]
code_files_ls = os.listdir(CodeFilePath)
code_files_ls.sort()
code_files_ls = [x for x in code_files_ls if "py" in x]
code_files_ls = code_files_ls[1:32]

# Loop over cde files
#--------------------
for i in range(0,len(code_files_ls)):

    file = code_files_ls[i]

    print(file)

    exec(open(CodeFilePath+file).read())
    print(f"✓ Loaded {file}")


print("="*60)
print("✅ All modules loaded successfully")
print("="*60 + "\n")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
