# RefusalPipeline Module
#-----------------------
# Main pipeline orchestrator for the complete refusal classification pipeline.
# Trains TWO independent classifiers: Refusal Classifier + Jailbreak Detector.
# 
# IMPROVEMENTS:
# - Full Config/Utils integration (NO HARDCODING!)
# - Uses safe_divide() from Utils for robust division
# - Uses get_timestamp() from Utils for consistent timestamps (already integrated)
# - Uses print_banner() from Utils for section headers
# - Improved comments and documentation
# All imports are in 01-Imports.py
###############################################################################


class RefusalPipeline:
    """
    Orchestrate the complete refusal classification pipeline with dual classifiers.
    
    This is the main pipeline that coordinates all steps from prompt generation
    through model training and evaluation. Trains two independent classifiers:
    
    1. **Refusal Classifier** (3-class): No Refusal, Hard Refusal, Soft Refusal
    2. **Jailbreak Detector** (2-class): Jailbreak Failed, Jailbreak Succeeded
    
    Pipeline Steps:
    1. Generate prompts (PromptGenerator)
    2. Collect LLM responses (ResponseCollector)  
    3. Clean data (DataCleaner)
    4. Label data with LLM judge (DataLabeler)
    5. Supplement jailbreak data if needed (WildJailbreak)
    6. Prepare datasets for both classifiers
    7. Train refusal classifier
    8. Train jailbreak detector
    9. Run analyses (cross-validation, per-model, confidence)
    10. Generate visualizations and reports
    
    All configuration comes from Config file (DATASET_CONFIG, TRAINING_CONFIG, etc.)
    """

    def __init__(self, api_keys: Dict, resume_from_checkpoint: bool = False):
        """
        Initialize pipeline.

        Args:
            api_keys: Dictionary with keys: 'openai', 'anthropic', 'google'
            resume_from_checkpoint: If True, resume from existing checkpoints
        """
        self.api_keys = api_keys
        self.resume_from_checkpoint = resume_from_checkpoint
        self.results = {}
        self.refusal_model = None
        self.jailbreak_model = None
        self.tokenizer = None
        
        self.run_timestamp = self._get_run_timestamp()

    def _get_run_timestamp(self) -> str:
        """
        Extract timestamp for this run from existing artifacts.
        
        Priority order:
        1. Prompts file (Step 1) - the source of truth for run timestamp
        2. Model checkpoints (Step 6+) - fallback if prompts deleted
        3. Current time - only if starting completely fresh
        
        This ensures timestamp consistency across resume scenarios.
        """
        
        # PRIORITY 1: Check for prompts file (Step 1 - earliest artifact)
        prompt_files = glob.glob(os.path.join(data_raw_path, "prompts_*.json"))
        if prompt_files:
            # Get most recent prompt file
            latest_prompt = max(prompt_files, key=os.path.getmtime)
            # Extract timestamp from filename: prompts_20241124_1430.json
            match = re.search(r'prompts_(\d{8}_\d{4})\.json', latest_prompt)
            if match:
                return match.group(1)
        
        # PRIORITY 2: Check for model files (Step 6+ - fallback)
        models_dirs = glob.glob(os.path.join(base_results_path, "*Models"))
        if models_dirs:
            models_dir = models_dirs[0]
            model_files = glob.glob(os.path.join(models_dir, "*.pt")) + glob.glob(os.path.join(models_dir, "*.pth"))
            
            if model_files:
                # Extract timestamps from ALL files and sort by timestamp value
                timestamps = []
                for filepath in model_files:
                    match = re.search(r'_(\d{8}_\d{4})_', filepath)
                    if match:
                        timestamps.append(match.group(1))
                
                if timestamps:
                    # Return the most recent timestamp (highest value)
                    return max(timestamps)
        
        # PRIORITY 3: Generate new timestamp (completely fresh run)
        return datetime.now().strftime("%Y%m%d_%H%M")
    
    
    def detect_available_data(self) -> Dict[str, Dict]:
        """
        Detect available intermediate data files and checkpoints.

        Returns:
            Dictionary mapping step names to available data/checkpoint info
        """
        available = {}

        # Step 1: Generated prompts
        prompt_files = glob.glob(os.path.join(data_raw_path, "prompts_*.json"))
        if prompt_files:
            latest = max(prompt_files, key=os.path.getmtime)
            available['prompts'] = {
                'path': latest,
                'basename': os.path.basename(latest),
                'age_hours': (time.time() - os.path.getmtime(latest)) / 3600,
                'step': 1
            }

        # Step 2: Response collection
        response_files = glob.glob(os.path.join(data_responses_path, "responses_*.pkl"))
        if response_files:
            latest = max(response_files, key=os.path.getmtime)
            available['responses'] = {
                'path': latest,
                'basename': os.path.basename(latest),
                'age_hours': (time.time() - os.path.getmtime(latest)) / 3600,
                'step': 2
            }

        # Step 3: Cleaned data
        cleaned_files = glob.glob(os.path.join(data_processed_path, "cleaned_responses_*.pkl"))
        if cleaned_files:
            latest = max(cleaned_files, key=os.path.getmtime)
            available['cleaned'] = {
                'path': latest,
                'basename': os.path.basename(latest),
                'age_hours': (time.time() - os.path.getmtime(latest)) / 3600,
                'step': 3
            }

        # Step 4: Labeled data
        labeled_files = glob.glob(os.path.join(data_processed_path, "labeled_responses_*.pkl"))
        if labeled_files:
            latest = max(labeled_files, key=os.path.getmtime)
            available['labeled'] = {
                'path': latest,
                'basename': os.path.basename(latest),
                'age_hours': (time.time() - os.path.getmtime(latest)) / 3600,
                'step': 4
            }

        # Step 5: Train/val/test splits
        train_files = glob.glob(os.path.join(data_splits_path, "train_*.pkl"))
        val_files = glob.glob(os.path.join(data_splits_path, "val_*.pkl"))
        test_files = glob.glob(os.path.join(data_splits_path, "test_*.pkl"))
        if train_files and val_files and test_files:
            latest_train = max(train_files, key=os.path.getmtime)
            available['splits'] = {
                'path': latest_train,  # Use train file as reference
                'basename': f"{len(train_files)} split files",
                'age_hours': (time.time() - os.path.getmtime(latest_train)) / 3600,
                'step': 5
            }

        # Step 6: Trained refusal classifier
        refusal_model_files = glob.glob(os.path.join(models_path, "*refusal*_best.pt"))
        if refusal_model_files:
            latest = max(refusal_model_files, key=os.path.getmtime)
            available['refusal_model'] = {
                'path': latest,
                'basename': os.path.basename(latest),
                'age_hours': (time.time() - os.path.getmtime(latest)) / 3600,
                'step': 6
            }

        # Step 7: Trained jailbreak detector
        jailbreak_model_files = glob.glob(os.path.join(models_path, "*jailbreak*_best.pt"))
        if jailbreak_model_files:
            latest = max(jailbreak_model_files, key=os.path.getmtime)
            available['jailbreak_model'] = {
                'path': latest,
                'basename': os.path.basename(latest),
                'age_hours': (time.time() - os.path.getmtime(latest)) / 3600,
                'step': 7
            }

        # Step 8: Adversarial testing results (NEW!)
        adversarial_files = glob.glob(os.path.join(analysis_results_path, "adversarial_testing_*.json"))
        if adversarial_files:
            latest = max(adversarial_files, key=os.path.getmtime)
            available['adversarial_results'] = {
                'path': latest,
                'basename': os.path.basename(latest),
                'age_hours': (time.time() - os.path.getmtime(latest)) / 3600,
                'step': 8
            }

        # Step 9: Analysis results (MOVED from Step 8)
        analysis_files = glob.glob(os.path.join(analysis_results_path, "*_analysis_*.json"))
        # Exclude adversarial testing files (they're Step 8 now)
        analysis_files = [f for f in analysis_files if 'adversarial_testing' not in f]
        if analysis_files:
            latest = max(analysis_files, key=os.path.getmtime)
            available['analysis_results'] = {
                'path': latest,
                'basename': f"{len(analysis_files)} analysis files",
                'age_hours': (time.time() - os.path.getmtime(latest)) / 3600,
                'step': 9
            }

        # Step 10: Visualizations (MOVED from Step 9)
        viz_files = glob.glob(os.path.join(reporting_viz_path, "*.png"))
        viz_files += glob.glob(os.path.join(reporting_viz_path, "**/*.png"), recursive=True)
        if viz_files:
            latest = max(viz_files, key=os.path.getmtime)
            available['visualizations'] = {
                'path': latest,
                'basename': f"{len(viz_files)} visualization files",
                'age_hours': (time.time() - os.path.getmtime(latest)) / 3600,
                'step': 10
            }

        # Step 11: Reports (MOVED from Step 10)
        report_files = glob.glob(os.path.join(reports_path, "*.pdf"))
        if report_files:
            latest = max(report_files, key=os.path.getmtime)
            available['reports'] = {
                'path': latest,
                'basename': f"{len(report_files)} PDF reports",
                'age_hours': (time.time() - os.path.getmtime(latest)) / 3600,
                'step': 11
            }

        return available


    def load_data_for_step(self, step: int) -> pd.DataFrame:
        """
        Load appropriate data for starting from a specific step.

        Args:
            step: Step number to start from (1-10)

        Returns:
            DataFrame with data needed for that step
        """
        if step <= 1:
            return None  # Start fresh

        available = self.detect_available_data()

        # For step 2 (collect responses), we need prompts - start fresh
        if step == 2:
            return None

        # For step 3 (clean data), load responses
        if step == 3:
            if 'responses' in available:
                print(f"📂 Loading responses from: {available['responses']['basename']}")
                return pd.read_pickle(available['responses']['path'])
            else:
                raise FileNotFoundError("No response data found. Please start from Step 2 (Collect Responses)")

        # For step 4 (label data), load cleaned responses
        if step == 4:
            if 'cleaned' in available:
                print(f"📂 Loading cleaned data from: {available['cleaned']['basename']}")
                return pd.read_pickle(available['cleaned']['path'])
            elif 'responses' in available:
                print(f"📂 No cleaned data found, loading responses from: {available['responses']['basename']}")
                print("   Will clean data before labeling...")
                return pd.read_pickle(available['responses']['path'])
            else:
                raise FileNotFoundError("No response or cleaned data found. Please start from Step 2")

        # For step 5+ (prepare datasets, training), load labeled data
        if step >= 5:
            if 'labeled' in available:
                print(f"📂 Loading labeled data from: {available['labeled']['basename']}")
                return pd.read_pickle(available['labeled']['path'])
            else:
                raise FileNotFoundError("No labeled data found. Please start from Step 4 (Label Data)")

        return None

    def load_trained_models(self) -> Tuple[Dict, Dict]:
        """
        Load pre-trained refusal and jailbreak models from disk.
        
        Returns:
            Tuple of (refusal_cv_results, jailbreak_cv_results) dictionaries
            containing loaded models and necessary metadata
        
        Raises:
            FileNotFoundError: If either model is missing
        """
        available = self.detect_available_data()
        
        # Check if both models exist
        if 'refusal_model' not in available:
            raise FileNotFoundError(
                "Refusal model not found. Cannot proceed with analysis.\n"
                "   Please start from Step 6 (Train Refusal Classifier)"
            )
        
        if 'jailbreak_model' not in available:
            raise FileNotFoundError(
                "Jailbreak model not found. Cannot proceed with analysis.\n"
                "   Please start from Step 7 (Train Jailbreak Detector)"
            )
        
        print_banner("LOADING PRE-TRAINED MODELS", width=60, char="─")
        
        # Initialize tokenizer if not already done
        if self.tokenizer is None:
            print("📝 Initializing tokenizer...")
            self.tokenizer = RobertaTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
        
        # ═════════════════════════════════════════════════════════════════════
        # LOAD REFUSAL CLASSIFIER
        # ═════════════════════════════════════════════════════════════════════
        
        refusal_model_path = available['refusal_model']['path']
        print(f"\n📂 Loading refusal classifier:")
        print(f"   Path: {os.path.basename(refusal_model_path)}")
        
        # Initialize refusal model
        self.refusal_model = RefusalClassifier(
            num_classes=MODEL_CONFIG['num_classes'],
            dropout=MODEL_CONFIG['dropout']
        ).to(DEVICE)
        
        # Load checkpoint
        checkpoint = safe_load_checkpoint(refusal_model_path, DEVICE)
        self.refusal_model.load_state_dict(checkpoint['model_state_dict'])
        self.refusal_model.eval()
        
        
        # Create refusal_cv_results structure (compatible with existing code)
        refusal_cv_results = {
            'final_model_path': refusal_model_path,
            'best_val_f1': checkpoint.get('best_val_f1', 0.0),
            'best_epoch': checkpoint.get('epoch', 0),
            'history': checkpoint.get('history', {}),
            'split_info': checkpoint.get('split_info', {}),  # FIX: Load split sizes
            'model_loaded': True,
            'loaded_from_checkpoint': True
        }
        
        print(f"   ✓ Loaded successfully (Best F1: {refusal_cv_results['best_val_f1']:.4f}, Epoch: {refusal_cv_results['best_epoch']})")
        if refusal_cv_results['split_info']:
            split_info = refusal_cv_results['split_info']
            print(f"   ✓ Split info: train_val={split_info.get('train_val_size', 'N/A')}, test={split_info.get('test_size', 'N/A')}")

        
        # ═════════════════════════════════════════════════════════════════════
        # LOAD JAILBREAK DETECTOR
        # ═════════════════════════════════════════════════════════════════════
        
        jailbreak_model_path = available['jailbreak_model']['path']
        print(f"\n📂 Loading jailbreak detector:")
        print(f"   Path: {os.path.basename(jailbreak_model_path)}")
        
        # Initialize jailbreak model
        self.jailbreak_model = JailbreakClassifier(
            num_classes=2,  # Binary classification
            dropout=MODEL_CONFIG['dropout']
        ).to(DEVICE)
        
        # Load checkpoint
        checkpoint = safe_load_checkpoint(jailbreak_model_path, DEVICE)
        self.jailbreak_model.load_state_dict(checkpoint['model_state_dict'])
        self.jailbreak_model.eval()
        
        
        
        # Create jailbreak_cv_results structure (compatible with existing code)
        jailbreak_cv_results = {
            'final_model_path': jailbreak_model_path,
            'best_val_f1': checkpoint.get('best_val_f1', 0.0),
            'best_epoch': checkpoint.get('epoch', 0),
            'history': checkpoint.get('history', {}),
            'split_info': checkpoint.get('split_info', {}),  # FIX: Load split sizes
            'model_loaded': True,
            'loaded_from_checkpoint': True
        }
        
        print(f"   ✓ Loaded successfully (Best F1: {jailbreak_cv_results['best_val_f1']:.4f}, Epoch: {jailbreak_cv_results['best_epoch']})")
        if jailbreak_cv_results['split_info']:
            split_info = jailbreak_cv_results['split_info']
            print(f"   ✓ Split info: train_val={split_info.get('train_val_size', 'N/A')}, test={split_info.get('test_size', 'N/A')}")
       
        
        print("─" * 60)
        print("✅ Both models loaded successfully")
        print("─" * 60 + "\n")
        
        return refusal_cv_results, jailbreak_cv_results
    
    
    def _save_step_checkpoint(self, step_number: int):
        """Save pipeline checkpoint after step completes."""
        CheckpointManager.save_pipeline_checkpoint(
            step_number=step_number,
            experiment_name=EXPERIMENT_CONFIG['experiment_name'],
            timestamp=self.run_timestamp
        )
    
    def run_partial_pipeline(self, start_step: int = 1):
        """
        Execute pipeline starting from a specific step.

        Args:
            start_step: Step number to start from (1-10)
                1: Generate Prompts
                2: Collect Responses
                3: Clean Data
                4: Label Data
                5: Prepare Datasets
                6: Train Refusal Classifier
                7: Train Jailbreak Detector
                8: Run Analyses
                9: Generate Visualizations
                10: Generate Reports
        """

# =============================================================================
#         # CRITICAL FIX: When resuming from Step 8+, extract timestamp from existing models
#         # WHY: Steps 8-10 analyze existing models, so should use model's timestamp, not current time
#         if start_step >= 8:
#             # Extract timestamp from existing model files
#             try:
#                 refusal_model_pattern = os.path.join(models_path, "*_refusal_best.pt")
#                 refusal_models = sorted(glob.glob(refusal_model_pattern), key=os.path.getmtime, reverse=True)
#                 
#                 if refusal_models:
#                     model_filename = os.path.basename(refusal_models[0])
#                     print(f"🔍 DEBUG: Extracting timestamp from: {model_filename}")
#                     parts = model_filename.split('_')
#                     
#                     # Find timestamp pattern (YYYYMMDD_HHMM)
#                     for i in range(len(parts) - 1):
#                         if len(parts[i]) == 8 and parts[i].isdigit() and len(parts[i+1]) == 4 and parts[i+1].isdigit():
#                             self.run_timestamp = f"{parts[i]}_{parts[i+1]}"
#                             print(f"✓ Extracted timestamp from model: {self.run_timestamp}")
#                             break
#                     else:
#                         # Fallback if pattern not found
#                         self.run_timestamp = get_timestamp('file')
#                         print(f"⚠️  Could not extract timestamp, using current: {self.run_timestamp}")
#                 else:
#                     # No models found
#                     self.run_timestamp = get_timestamp('file')
#                     print(f"⚠️  No models found, using current timestamp: {self.run_timestamp}")
#             except Exception as e:
#                 self.run_timestamp = get_timestamp('file')
#                 print(f"⚠️  Error extracting timestamp: {e}, using current: {self.run_timestamp}")
#         else:
#             # For steps < 8 (training steps), generate new timestamp
#             self.run_timestamp = get_timestamp('file')
#             print(f"🆕 Generated new timestamp for training: {self.run_timestamp}")
# =============================================================================

# =============================================================================
#         if start_step > 1:
#             # Resuming - load checkpoint
#             print(f"\n{'='*60}")
#             print(f"RESUMING FROM STEP {start_step}")
#             print(f"{'='*60}")
#             
#             checkpoint = CheckpointManager.load_pipeline_checkpoint()
#             
#             if checkpoint:
#                 # Restore from checkpoint
#                 EXPERIMENT_CONFIG['experiment_name'] = checkpoint['experiment_name']
#                 self.run_timestamp = checkpoint['timestamp']
#                 
#                 print(f"Found checkpoint:")
#                 print(f"  Last step: {checkpoint['step_completed']}")
#                 print(f"  Experiment: {checkpoint['experiment_name']}")
#                 print(f"  ✓ RESTORED timestamp: {checkpoint['timestamp']}")
#             else:
#                 # No checkpoint - fallback to model extraction
#                 print(f"⚠️  No checkpoint found, extracting from model...")
#                 
#                 try:
#                     refusal_model_pattern = os.path.join(models_path, "*_refusal_best.pt")
#                     refusal_models = sorted(glob.glob(refusal_model_pattern), key=os.path.getmtime, reverse=True)
#                     
#                     if refusal_models:
#                         model_filename = os.path.basename(refusal_models[0])
#                         parts = model_filename.split('_')
#                         
#                         # Find timestamp pattern (YYYYMMDD_HHMM)
#                         for i in range(len(parts) - 1):
#                             if len(parts[i]) == 8 and parts[i].isdigit() and len(parts[i+1]) == 4 and parts[i+1].isdigit():
#                                 model_timestamp = f"{parts[i]}_{parts[i+1]}"
#                                 self.run_timestamp = model_timestamp
#                                 
#                                 # Restore experiment name from model timestamp
#                                 EXPERIMENT_CONFIG['experiment_name'] = f"dual_RoBERTa_classifier_{model_timestamp}"
#                                 
#                                 print(f"  ✓ Extracted from model: {model_filename}")
#                                 print(f"  ✓ Timestamp: {model_timestamp}")
#                                 break
#                         else:
#                             # Pattern not found
#                             self.run_timestamp = get_timestamp('file')
#                             print(f"  ⚠️  Using current timestamp: {self.run_timestamp}")
#                     else:
#                         # No models found
#                         self.run_timestamp = get_timestamp('file')
#                         print(f"  ⚠️  No models found, using current: {self.run_timestamp}")
#                 except Exception as e:
#                     self.run_timestamp = get_timestamp('file')
#                     print(f"  ⚠️  Error: {e}, using current: {self.run_timestamp}")
#             
#             print(f"{'='*60}\n")
#         else:
#             # Step 1: Fresh start
#             self.run_timestamp = get_timestamp('file')
#             print(f"🆕 Starting fresh experiment with timestamp: {self.run_timestamp}")
# 
# =============================================================================

        print_banner(f"REFUSAL CLASSIFIER - PARTIAL PIPELINE (START: STEP {start_step})", width=60)
        print(f"Experiment: {EXPERIMENT_CONFIG['experiment_name']}")
        print(f"Run Timestamp: {self.run_timestamp}")
        print(f"Classifier 1: Refusal Classification (3 classes)")
        print(f"Classifier 2: Jailbreak Detection (2 classes)")
        print("="*60 + "\n")

# =============================================================================
#         # Load data if starting from later step
#         prompts = None
#         responses_df = None
#         cleaned_df = None
#         labeled_df = None
#         datasets = None
# =============================================================================
    
        
        # ═════════════════════════════════════════════════════════════════════
        # STEP 1: GENERATE PROMPTS
        # ═════════════════════════════════════════════════════════════════════
        prompts = None
        
        if start_step == 1:
            prompts = self.generate_prompts()
            self._save_step_checkpoint(1)
        
        # ═════════════════════════════════════════════════════════════════════
        # STEP 2: COLLECT RESPONSES
        # ═════════════════════════════════════════════════════════════════════
        responses_df = None
        
        if start_step <= 2:
            # Need prompts for Step 2
            if prompts is None:
                # Load prompts from saved file
                available = self.detect_available_data()
                if 'prompts' not in available:
                    raise FileNotFoundError("No saved prompts found. Please start from Step 1")
                
                with open(available['prompts']['path'], 'r') as f:
                    prompts = json.load(f)
                print(f"📂 Loaded prompts from: {available['prompts']['basename']}")
            
            responses_df = self.collect_responses(prompts)
            self._save_step_checkpoint(2)
        else:
            # Load existing responses
            responses_df = self.load_data_for_step(3)
        
        # ═════════════════════════════════════════════════════════════════════
        # STEP 3: CLEAN DATA
        # ═════════════════════════════════════════════════════════════════════
        cleaned_df = None
        
        if start_step <= 3:
            if responses_df is None:
                responses_df = self.load_data_for_step(3)
            cleaned_df = self.clean_data(responses_df)
            self._save_step_checkpoint(3)
        else:
            cleaned_df = self.load_data_for_step(4)
        
        # ═════════════════════════════════════════════════════════════════════
        # STEP 4: LABEL DATA
        # ═════════════════════════════════════════════════════════════════════
        labeled_df = None
        labeled_df_augmented = None
        
        if start_step <= 4:
            if cleaned_df is None:
                cleaned_df = self.load_data_for_step(4)
            labeled_df = self.label_data(cleaned_df)
            labeled_df_augmented = self.prepare_jailbreak_training_data(labeled_df)
            self._save_step_checkpoint(4)
        else:
            labeled_df = self.load_data_for_step(5)
            labeled_df_augmented = labeled_df
        
        # ═════════════════════════════════════════════════════════════════════
        # STEP 5: PREPARE DATASETS
        # ═════════════════════════════════════════════════════════════════════
        datasets = None
        
        if start_step <= 5:
            if labeled_df_augmented is None:
                labeled_df = self.load_data_for_step(5)
                labeled_df_augmented = labeled_df
            datasets = self.prepare_datasets(labeled_df_augmented)
            self._save_step_checkpoint(5)
        
        # ═════════════════════════════════════════════════════════════════════
        # STEP 6 & 7: TRAINING (WITH SMART MODEL DETECTION)
        # ═════════════════════════════════════════════════════════════════════
        
        refusal_cv_results = None
        jailbreak_cv_results = None
        test_df = None
        
        # Check if trained models already exist
        available = self.detect_available_data()
        models_exist = ('refusal_model' in available and 'jailbreak_model' in available)
        
        # STEP 6: Refusal Classifier
        if start_step <= 6:
            if models_exist and start_step >= 6:
                # Models exist and starting at/after Step 6: Load them
                print("\n" + "="*60)
                print("⚠️  TRAINED MODELS DETECTED")
                print("="*60)
                print(f"  Refusal model: {available['refusal_model']['basename']}")
                print(f"  Jailbreak model: {available['jailbreak_model']['basename']}")
                print("  Loading existing models instead of retraining...")
                print("="*60 + "\n")
                
                try:
                    refusal_cv_results, jailbreak_cv_results = self.load_trained_models()
                except FileNotFoundError as e:
                    print(f"❌ ERROR: {e}")
                    return
            else:
                # Train refusal classifier
                if datasets is None:
                    if labeled_df_augmented is None:
                        labeled_df = self.load_data_for_step(5)
                        labeled_df_augmented = labeled_df
                    datasets = self.prepare_datasets(labeled_df_augmented)
                
                refusal_cv_results = self.train_refusal_classifier(
                    datasets['refusal']['full_dataset']
                )
                self._save_step_checkpoint(6)
                
                # CRITICAL: Memory cleanup before training second classifier
                gc.collect()
                if DEVICE.type == 'mps':
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        if hasattr(torch.mps, 'synchronize'):
                            torch.mps.synchronize()
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                    gc.collect()
                    print("🧹 MPS memory cleanup between classifiers complete")
                elif DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                    print("🧹 CUDA memory cleanup between classifiers complete")
                    
        
        # STEP 7: Jailbreak Detector  
        if start_step <= 7:
            if jailbreak_cv_results is None:
                if models_exist and start_step >= 7 and 'jailbreak_model' in available:
                    # Jailbreak model exists, load it
                    print("\n" + "="*60)
                    print("⚠️  TRAINED JAILBREAK MODEL DETECTED")
                    print("="*60)
                    print(f"  Model: {available['jailbreak_model']['basename']}")
                    print("  Loading existing model...")
                    print("="*60 + "\n")
                    
                    try:
                        if refusal_cv_results and 'loaded_from_checkpoint' not in refusal_cv_results:
                            # Refusal was just trained, only load jailbreak
                            _, jailbreak_cv_results = self.load_trained_models()
                        else:
                            # Load both models
                            refusal_cv_results, jailbreak_cv_results = self.load_trained_models()
                    except FileNotFoundError as e:
                        print(f"❌ ERROR: {e}")
                        return
                else:
                    # Train jailbreak detector
                    if datasets is None:
                        if labeled_df_augmented is None:
                            labeled_df = self.load_data_for_step(5)
                            labeled_df_augmented = labeled_df
                        datasets = self.prepare_datasets(labeled_df_augmented)
                    
                    jailbreak_cv_results = self.train_jailbreak_detector(
                        datasets['jailbreak']['full_dataset']
                    )
                    self._save_step_checkpoint(7)
        
        
        # ═════════════════════════════════════════════════════════════════════
        # STEP 8: ADVERSARIAL TESTING
        # ═════════════════════════════════════════════════════════════════════
        
        adversarial_results = None
        
        # Load models if starting from Step 8+
        if start_step >= 8:
            print("\n" + "="*60)
            print("📦 STEP 8+ REQUIRES TRAINED MODELS")
            print("="*60)
            print("  Checking for pre-trained models...")
            
            # Load CV results if not already loaded
            if refusal_cv_results is None or jailbreak_cv_results is None:
                try:
                    refusal_cv_results, jailbreak_cv_results = self.load_trained_models()
                except FileNotFoundError as e:
                    print(f"\n❌ ERROR: {e}")
                    print("\n💡 SOLUTION:")
                    print("   Option 1: Start from Step 6 to train both models")
                    print("   Option 2: Start from Step 7 if refusal model exists")
                    print("\n   Exiting pipeline...")
                    return
            
            # Load actual model objects
            if self.refusal_model is None:
                print("  Loading refusal classifier model...")
                refusal_checkpoint = self._find_latest_checkpoint('refusal')
                if refusal_checkpoint:
                    self.refusal_model = self._load_model_from_checkpoint(
                        refusal_checkpoint, 
                        MODEL_CONFIG, 
                        'refusal'
                    )
                    print(f"  ✓ Loaded refusal model from: {os.path.basename(refusal_checkpoint)}")
                else:
                    print("  ❌ No refusal checkpoint found!")
                    return
            
            if self.jailbreak_model is None:
                print("  Loading jailbreak detector model...")
                jailbreak_checkpoint = self._find_latest_checkpoint('jailbreak')
                if jailbreak_checkpoint:
                    self.jailbreak_model = self._load_model_from_checkpoint(
                        jailbreak_checkpoint,
                        JAILBREAK_CONFIG,
                        'jailbreak'
                    )
                    print(f"  ✓ Loaded jailbreak model from: {os.path.basename(jailbreak_checkpoint)}")
                else:
                    print("  ❌ No jailbreak checkpoint found!")
                    return
            
            # Extract timestamp from model filename
            try:
                refusal_model_pattern = os.path.join(models_path, "*_refusal_best.pt")
                refusal_models = sorted(glob.glob(refusal_model_pattern), key=os.path.getmtime, reverse=True)
                
                if refusal_models:
                    model_filename = os.path.basename(refusal_models[0])
                    parts = model_filename.split('_')
                    
                    for i in range(len(parts) - 1):
                        if len(parts[i]) == 8 and parts[i].isdigit() and len(parts[i+1]) == 4 and parts[i+1].isdigit():
                            self.run_timestamp = f"{parts[i]}_{parts[i+1]}"
                            print(f"  ✓ Using model timestamp: {self.run_timestamp}")
                            break
                    else:
                        self.run_timestamp = get_timestamp('file')
                        print(f"  ⚠️  Could not extract timestamp, using current: {self.run_timestamp}")
            except Exception as e:
                self.run_timestamp = get_timestamp('file')
                print(f"  ⚠️  Error extracting timestamp: {e}")
            
            # Load labeled_df_augmented if needed
            if labeled_df_augmented is None:
                labeled_df = self.load_data_for_step(5)
                labeled_df_augmented = labeled_df
        
        # Create test_df from CV results (needed for Steps 8+)
        if start_step <= 8:
            if refusal_cv_results and jailbreak_cv_results:
                test_df = self._create_test_df_from_cv_results(
                    refusal_cv_results,
                    jailbreak_cv_results,
                    labeled_df_augmented
                )
            else:
                # Fallback: load from datasets if CV results not available
                test_df = datasets['refusal']['test_df'] if datasets else None
        
        # STEP 8: Run Adversarial Testing (Paraphrasing)
        if start_step <= 8:
            if test_df is not None and self.refusal_model is not None:
                adversarial_results = self.run_adversarial_testing(test_df)
                
                # Only save checkpoint if step 8 actually ran
                self._save_step_checkpoint(8)
            
            else:
                print("⚠️  No test data or model available for adversarial testing")
                print("⚠️  Skipping step 8 - checkpoint NOT saved")
                adversarial_results = None
        
        # ═════════════════════════════════════════════════════════════════════
        # STEP 9-11: LOAD PREREQUISITES IF RESUMING FROM THESE STEPS
        # ═════════════════════════════════════════════════════════════════════
        
        # If starting from step 9+, ensure we have all prerequisites loaded
        if start_step >= 9 and start_step <= 11:
            print("\n" + "="*60)
            print(f"📦 RESUMING FROM STEP {start_step} - LOADING PREREQUISITES")
            print("="*60)
            
            # Load test_df if not available (needed for steps 9-11)
            if test_df is None:
                if refusal_cv_results and jailbreak_cv_results:
                    if labeled_df_augmented is None:
                        labeled_df = self.load_data_for_step(5)
                        labeled_df_augmented = labeled_df
                    test_df = self._create_test_df_from_cv_results(
                        refusal_cv_results,
                        jailbreak_cv_results,
                        labeled_df_augmented
                    )
                    print(f"  ✓ Loaded test_df with {len(test_df)} samples")
            
            
            # ═════════════════════════════════════════════════════════════════════
            # STEP 10-11: LOAD PREREQUISITES IF RESUMING FROM THESE STEPS
            # ═════════════════════════════════════════════════════════════════════
            
            # If starting from step 10+, ensure we have analysis_results loaded
            if start_step >= 10:
                print("\n" + "="*60)
                print(f"📦 RESUMING FROM STEP {start_step} - LOADING PREREQUISITES")
                print("="*60)
                
                analysis_results = None
                
                # Load analysis results if not available (needed for steps 10-11)
                if analysis_results is None:
                    analysis_results_files = glob.glob(os.path.join(
                        analysis_results_path, 
                        "refusal_jailbreak_analysis_results_complete_*.json"
                    ))
                    if analysis_results_files:
                        latest_analysis = max(analysis_results_files, key=os.path.getmtime)
                        print(f"  📂 Loading analysis results: {os.path.basename(latest_analysis)}")
                        with open(latest_analysis, 'r') as f:
                            analysis_results = json.load(f)
                        print("  ✓ Analysis results loaded")
                        
                        # CRITICAL FIX: Auto-load missing adversarial/correlation data
                        if not analysis_results.get('adversarial'):
                            print("  ⚠️  Adversarial data missing - attempting to load...")
                            adversarial_files = glob.glob(os.path.join(analysis_results_path, "adversarial_testing_*.json"))
                            if adversarial_files:
                                latest_adv = max(adversarial_files, key=os.path.getmtime)
                                with open(latest_adv, 'r') as f:
                                    analysis_results['adversarial'] = json.load(f)
                                print(f"  ✓ Loaded adversarial data from {os.path.basename(latest_adv)}")
                        
                        if not analysis_results.get('correlation') or len(analysis_results.get('correlation', {})) == 0:
                            print("  ⚠️  Correlation data missing - attempting to load...")
                            correlation_files = glob.glob(os.path.join(analysis_results_path, "correlation_correlation_analysis_*.pkl"))
                            if correlation_files:
                                latest_corr = max(correlation_files, key=os.path.getmtime)
                                with open(latest_corr, 'rb') as f:
                                    analysis_results['correlation'] = pickle.load(f)
                                print(f"  ✓ Loaded correlation data from {os.path.basename(latest_corr)}")
                    else:
                        print("\n  ❌ ERROR: No analysis results found!")
                        print("  💡 SOLUTION: Start from Step 9")
                        return
                
                # Regenerate figures if starting from step 11
                if start_step == 11:
                    if refusal_cv_results and jailbreak_cv_results and analysis_results:
                        print("  📊 Regenerating visualizations...")
                        refusal_history = refusal_cv_results.get('history', {})
                        jailbreak_history = jailbreak_cv_results.get('history', {})
                        figures = self.generate_visualizations(refusal_history, jailbreak_history, analysis_results)
                        print("  ✓ Visualizations regenerated")
                
                print("="*60 + "\n")
        
            

# =============================================================================
#             # Load analysis results if starting from step 10 or 11
#             if start_step >= 10:
#                  analysis_results_files = glob.glob(os.path.join(
#                      analysis_results_path, 
#                     "refusal_jailbreak_analysis_results_complete_*.json"
#                 ))
#                 
#                 
#                 
#             if analysis_results_files:
#                     latest_analysis = max(analysis_results_files, key=os.path.getmtime)
#                     print(f"  📂 Loading analysis results: {os.path.basename(latest_analysis)}")
#                     with open(latest_analysis, 'r') as f:
#                         analysis_results = json.load(f)
#                     print("  ✓ Analysis results loaded")
#                     
#                     
#                     # CRITICAL FIX: Auto-load missing adversarial/correlation data
#                     # If these are empty/None in complete JSON, try loading separate files
#                     if not analysis_results.get('adversarial'):
#                         print("  ⚠️  Adversarial data missing - attempting to load from separate file...")
#                         adversarial_files = glob.glob(os.path.join(analysis_results_path, "adversarial_testing_*.json"))
#                         if adversarial_files:
#                             latest_adv = max(adversarial_files, key=os.path.getmtime)
#                             with open(latest_adv, 'r') as f:
#                                 analysis_results['adversarial'] = json.load(f)
#                             print(f"  ✓ Loaded adversarial data from {os.path.basename(latest_adv)}")
#                         else:
#                             print("  ⚠️  No adversarial data found")
#                     
#                     if not analysis_results.get('correlation') or len(analysis_results.get('correlation', {})) == 0:
#                         print("  ⚠️  Correlation data missing - attempting to load from separate file...")
#                         # FIXED: Search in analysis_results_path, not correlation_viz_path!
#                         correlation_files = glob.glob(os.path.join(analysis_results_path, "correlation_correlation_analysis_*.pkl"))
#                         if correlation_files:
#                             latest_corr = max(correlation_files, key=os.path.getmtime)
#                             with open(latest_corr, 'rb') as f:
#                                 analysis_results['correlation'] = pickle.load(f)
#                             print(f"  ✓ Loaded correlation data from {os.path.basename(latest_corr)}")
#                         else:
#                             print("  ⚠️  No correlation data found")
#                     
#             else:
#                     print("\n  ❌ ERROR: No analysis results found!")
#                     print("  💡 SOLUTION: Start from Step 9 to generate analysis results")
#                     return
#             
#             
#             # Load/regenerate figures if starting from step 11
#             if start_step == 11:
#                 # Must regenerate visualizations (figures aren't saved as objects)
#                 if analysis_results and refusal_cv_results and jailbreak_cv_results:
#                     print("  📊 Regenerating visualizations for report generation...")
#                     refusal_history = refusal_cv_results.get('history', {})
#                     jailbreak_history = jailbreak_cv_results.get('history', {})
#                     figures = self.generate_visualizations(
#                         refusal_history, 
#                         jailbreak_history, 
#                         analysis_results
#                     )
#                     print("  ✓ Visualizations regenerated")
#                 else:
#                     print("\n  ❌ ERROR: Cannot regenerate visualizations!")
#                     print("  💡 SOLUTION: Start from Step 9 to generate full analysis")
#                     return
#             
#             print("="*60 + "\n")
# 
# =============================================================================
        

        # ═════════════════════════════════════════════════════════════════════
        # STEP 9: ANALYSES (LOADS ADVERSARIAL RESULTS)
        # ═════════════════════════════════════════════════════════════════════
        
        # Initialize if not already loaded from resume block
        if start_step < 9:
            analysis_results = None
        elif start_step == 9:
            analysis_results = None  # Will be generated fresh
        # else: already loaded in resume block above
        
        if start_step <= 9:
            # Load test_df if not already available
            if test_df is None:
                if refusal_cv_results and jailbreak_cv_results:
                    if labeled_df_augmented is None:
                        labeled_df = self.load_data_for_step(5)
                        labeled_df_augmented = labeled_df
                    test_df = self._create_test_df_from_cv_results(
                        refusal_cv_results,
                        jailbreak_cv_results,
                        labeled_df_augmented
                    )
            
            if test_df is not None:
                # Load adversarial results if they weren't just generated
                if adversarial_results is None and 'adversarial_results' in available:
                    adversarial_results = self._load_adversarial_results()
                
                # Run analyses (with optional adversarial results)
                analysis_results = self.run_analyses(test_df, adversarial_results)
                
                self._save_step_checkpoint(9)
                
            else:
                print("⚠️  No test data available for analysis")
                analysis_results = None
        
        # ═════════════════════════════════════════════════════════════════════
        # STEP 10: VISUALIZATIONS
        # ═════════════════════════════════════════════════════════════════════
        
        # Initialize if not already loaded from resume block
        if start_step < 10:
            figures = None
        elif start_step == 10:
            figures = None  # Will be generated fresh
        # else: already loaded/generated in resume block above
        
        if start_step <= 10:
            if refusal_cv_results and jailbreak_cv_results and analysis_results:
                refusal_history = refusal_cv_results.get('history', {})
                jailbreak_history = jailbreak_cv_results.get('history', {})
                figures = self.generate_visualizations(refusal_history, jailbreak_history, analysis_results)
                
                self._save_step_checkpoint(10)
                
            else:
                print("⚠️  Skipping visualizations - missing required data")
                figures = None
        
        # ═════════════════════════════════════════════════════════════════════
        # STEP 11: REPORTS
        # ═════════════════════════════════════════════════════════════════════
        
        if start_step <= 11:
            if figures and refusal_cv_results and jailbreak_cv_results and analysis_results:
                self.generate_reports(refusal_cv_results, jailbreak_cv_results, analysis_results, figures)
                
                self._save_step_checkpoint(11)
                
            else:
                print("⚠️  Skipping report generation - missing required data")
        
        print("\n" + "="*60)
        print(f"✅ PARTIAL PIPELINE COMPLETE (Started from Step {start_step})")
        print("="*60)

# =============================================================================
#         # Execute pipeline from specified step
#         if start_step <= 1:
#             prompts = self.generate_prompts()
#             self._save_step_checkpoint(1)
# 
#         if start_step <= 2:
#             if start_step == 2:
#                 prompts = self.generate_prompts()  # Need prompts for collection
#             responses_df = self.collect_responses(prompts)
#             self._save_step_checkpoint(2)
#             
#         elif start_step >= 3:
#             responses_df = self.load_data_for_step(3)
#             
# 
#         if start_step <= 3:
#             cleaned_df = self.clean_data(responses_df)
#             self._save_step_checkpoint(3)
#             
#         elif start_step >= 4:
#             cleaned_df = self.load_data_for_step(4)
# 
#         if start_step <= 4:
#             labeled_df = self.label_data(cleaned_df)
#             labeled_df_augmented = self.prepare_jailbreak_training_data(labeled_df)
#             self._save_step_checkpoint(4)
#             
#         else:
#             # Starting from step 5+: Just load the data, NO step 4.5!
#             labeled_df = self.load_data_for_step(5)
#             labeled_df_augmented = labeled_df  # Use as-is (already processed in previous run)
# 
#         if start_step <= 5:
#             datasets = self.prepare_datasets(labeled_df_augmented)
#             self._save_step_checkpoint(5)
# 
#         # ═════════════════════════════════════════════════════════════════════
#         # STEP 6 & 7: TRAINING (WITH SMART MODEL DETECTION)
#         # ═════════════════════════════════════════════════════════════════════
#         
#         refusal_cv_results = None
#         jailbreak_cv_results = None
#         test_df = None
#         
#         # CRITICAL: Check if trained models already exist BEFORE training!
#         available = self.detect_available_data()
#         models_exist = ('refusal_model' in available and 'jailbreak_model' in available)
#         
#         # STEP 6: Refusal Classifier
#         if start_step <= 6:
#             if models_exist and start_step >= 6:
#                 # Models already exist! Load them instead of retraining
#                 print("\n" + "="*60)
#                 print("⚠️  TRAINED MODELS DETECTED")
#                 print("="*60)
#                 print(f"  Refusal model: {available['refusal_model']['basename']}")
#                 print(f"  Jailbreak model: {available['jailbreak_model']['basename']}")
#                 print("\n  These models were trained in a previous run.")
#                 print("  Loading existing models instead of retraining...")
#                 print("="*60 + "\n")
#                 
#                 try:
#                     refusal_cv_results, jailbreak_cv_results = self.load_trained_models()
#                 except FileNotFoundError as e:
#                     print(f"❌ ERROR: {e}")
#                     return
#             else:
#                 # No models exist OR explicitly starting from Step 6 → Train
#                 if datasets is None:
#                     # Load datasets if we skipped step 5
#                     datasets = self.prepare_datasets(labeled_df_augmented)
#                 refusal_cv_results = self.train_refusal_classifier(
#                     datasets['refusal']['full_dataset']
#                 )
#                 self._save_step_checkpoint(6)
#         
#         # STEP 7: Jailbreak Detector  
#         if start_step <= 7:
#             # Only train if we haven't already loaded models in Step 6
#             if jailbreak_cv_results is None:
#                 if models_exist and start_step >= 7 and 'jailbreak_model' in available:
#                     # Jailbreak model exists but wasn't loaded yet
#                     print("\n" + "="*60)
#                     print("⚠️  TRAINED JAILBREAK MODEL DETECTED")
#                     print("="*60)
#                     print(f"  Model: {available['jailbreak_model']['basename']}")
#                     print("  Loading existing model instead of retraining...")
#                     print("="*60 + "\n")
#                     
#                     try:
#                         # Load only jailbreak model if refusal was just trained
#                         if refusal_cv_results and 'loaded_from_checkpoint' not in refusal_cv_results:
#                             # Refusal was just trained, only load jailbreak
#                             _, jailbreak_cv_results = self.load_trained_models()
#                         else:
#                             # Load both (though refusal might be redundant)
#                             refusal_cv_results, jailbreak_cv_results = self.load_trained_models()
#                     except FileNotFoundError as e:
#                         print(f"❌ ERROR: {e}")
#                         return
#                 else:
#                     # No jailbreak model exists OR explicitly starting from Step 7 → Train
#                     if datasets is None:
#                         # Load datasets if we skipped step 5
#                         datasets = self.prepare_datasets(labeled_df_augmented)
#                     jailbreak_cv_results = self.train_jailbreak_detector(
#                         datasets['jailbreak']['full_dataset']
#                     )
#                     self._save_step_checkpoint(7)
#         
#         
#         # ═════════════════════════════════════════════════════════════════════
#         # STEP 8: ADVERSARIAL TESTING (NEW SEPARATE STEP!)
#         # ═════════════════════════════════════════════════════════════════════
#         
#         adversarial_results = None
#         
#         # CRITICAL: If starting from Step 8+, ensure models are loaded
#         if start_step >= 8:
#             print("\n" + "="*60)
#             print("📦 STEP 8+ REQUIRES TRAINED MODELS")
#             print("="*60)
#             print("  Checking for pre-trained models...")
#             
#             # Load CV results if not already loaded
#             if refusal_cv_results is None or jailbreak_cv_results is None:
#                 try:
#                     refusal_cv_results, jailbreak_cv_results = self.load_trained_models()
#                 except FileNotFoundError as e:
#                     print(f"\n❌ ERROR: {e}")
#                     print("\n💡 SOLUTION:")
#                     print("   Option 1: Start from Step 6 to train both models")
#                     print("   Option 2: Start from Step 7 if refusal model exists")
#                     print("\n   Exiting pipeline...")
#                     return
#             
#             # CRITICAL: Load actual model objects (not just CV results)
#             if self.refusal_model is None:
#                 print("  Loading refusal classifier model...")
#                 refusal_checkpoint = self._find_latest_checkpoint('refusal')
#                 if refusal_checkpoint:
#                     self.refusal_model = self._load_model_from_checkpoint(
#                         refusal_checkpoint, 
#                         MODEL_CONFIG, 
#                         'refusal'
#                     )
#                     print(f"  ✓ Loaded refusal model from: {os.path.basename(refusal_checkpoint)}")
#                 else:
#                     print("  ❌ No refusal checkpoint found!")
#                     return
#             
#             if self.jailbreak_model is None:
#                 print("  Loading jailbreak detector model...")
#                 jailbreak_checkpoint = self._find_latest_checkpoint('jailbreak')
#                 if jailbreak_checkpoint:
#                     self.jailbreak_model = self._load_model_from_checkpoint(
#                         jailbreak_checkpoint,
#                         JAILBREAK_CONFIG,
#                         'jailbreak'
#                     )
#                     print(f"  ✓ Loaded jailbreak model from: {os.path.basename(jailbreak_checkpoint)}")
#                 else:
#                     print("  ❌ No jailbreak checkpoint found!")
#                     return
#             
#             # Extract timestamp from model filename (your commented code was correct!)
#             try:
#                 refusal_model_pattern = os.path.join(models_path, "*_refusal_best.pt")
#                 refusal_models = sorted(glob.glob(refusal_model_pattern), key=os.path.getmtime, reverse=True)
#                 
#                 if refusal_models:
#                     model_filename = os.path.basename(refusal_models[0])
#                     parts = model_filename.split('_')
#                     
#                     # Find timestamp pattern (YYYYMMDD_HHMM)
#                     for i in range(len(parts) - 1):
#                         if len(parts[i]) == 8 and parts[i].isdigit() and len(parts[i+1]) == 4 and parts[i+1].isdigit():
#                             self.run_timestamp = f"{parts[i]}_{parts[i+1]}"
#                             print(f"  ✓ Using model timestamp: {self.run_timestamp}")
#                             break
#                     else:
#                         self.run_timestamp = get_timestamp('file')
#                         print(f"  ⚠️  Could not extract timestamp, using current: {self.run_timestamp}")
#             except Exception as e:
#                 self.run_timestamp = get_timestamp('file')
#                 print(f"  ⚠️  Error extracting timestamp: {e}")
#         
#         # Create test_df from CV results (needed for Steps 8+)
#         if start_step <= 8:
#             if refusal_cv_results and jailbreak_cv_results:
#                 test_df = self._create_test_df_from_cv_results(
#                     refusal_cv_results,
#                     jailbreak_cv_results,
#                     labeled_df_augmented
#                 )
#             else:
#                 # Fallback: load from datasets if CV results not available
#                 test_df = datasets['refusal']['test_df'] if datasets else None
#         
#         # STEP 8: Run Adversarial Testing (Paraphrasing)
#         if start_step <= 8:
#             if test_df is not None and self.refusal_model is not None:
#                 adversarial_results = self.run_adversarial_testing(test_df)
#             else:
#                 print("⚠️  No test data or model available for adversarial testing")
#                 adversarial_results = None
#             
#             self._save_step_checkpoint(8)
#         
#         # ═════════════════════════════════════════════════════════════════════
#         # STEP 9: ANALYSES (LOADS ADVERSARIAL RESULTS)
#         # ═════════════════════════════════════════════════════════════════════
#         
#         analysis_results = None
#         
#         if start_step <= 9:
#             # Load test_df if not already available
#             if test_df is None:
#                 if refusal_cv_results and jailbreak_cv_results:
#                     test_df = self._create_test_df_from_cv_results(
#                         refusal_cv_results,
#                         jailbreak_cv_results,
#                         labeled_df_augmented
#                     )
#             
#             if test_df is not None:
#                 # Load adversarial results if they weren't just generated
#                 if adversarial_results is None and 'adversarial_results' in available:
#                     adversarial_results = self._load_adversarial_results()
#                 
#                 # Run analyses (with optional adversarial results)
#                 analysis_results = self.run_analyses(test_df, adversarial_results)
#                 
#                 self._save_step_checkpoint(9)
#                 
#             else:
#                 print("⚠️  No test data available for analysis")
#                 analysis_results = None
#         
#         # ═════════════════════════════════════════════════════════════════════
#         # STEP 10: VISUALIZATIONS
#         # ═════════════════════════════════════════════════════════════════════
#         
#         figures = None
#         
#         if start_step <= 10:
#             if refusal_cv_results and jailbreak_cv_results and analysis_results:
#                 refusal_history = refusal_cv_results.get('history', {})
#                 jailbreak_history = jailbreak_cv_results.get('history', {})
#                 figures = self.generate_visualizations(refusal_history, jailbreak_history, analysis_results)
#                 
#                 self._save_step_checkpoint(10)
#                 
#             else:
#                 print("⚠️  Skipping visualizations - missing required data")
#                 figures = None
#         
#         # ═════════════════════════════════════════════════════════════════════
#         # STEP 11: REPORTS
#         # ═════════════════════════════════════════════════════════════════════
#         
#         if start_step <= 11:
#             if figures and refusal_cv_results and jailbreak_cv_results and analysis_results:
#                 self.generate_reports(refusal_cv_results, jailbreak_cv_results, analysis_results, figures)
#                 
#                 self._save_step_checkpoint(11)
#                 
#             else:
#                 print("⚠️  Skipping report generation - missing required data")
# 
#         print("\n" + "="*60)
#         print(f"✅ PARTIAL PIPELINE COMPLETE (Started from Step {start_step})")
#         print("="*60)
# 
# =============================================================================

    def run_full_pipeline(self):
        """Execute complete pipeline from start to finish."""
        # Generate single timestamp for this entire run
        self.run_timestamp = get_timestamp('file')

        print_banner("REFUSAL CLASSIFIER - FULL PIPELINE (DUAL CLASSIFIERS)", width=60)
        print(f"Experiment: {EXPERIMENT_CONFIG['experiment_name']}")
        print(f"Run Timestamp: {self.run_timestamp}")
        print(f"Classifier 1: Refusal Classification (3 classes)")
        print(f"Classifier 2: Jailbreak Detection (2 classes)")
        print("="*60 + "\n")

        # Step 1: Generate prompts
        prompts = self.generate_prompts()
        
        self._save_step_checkpoint(1)

        # Step 2: Collect responses
        responses_df = self.collect_responses(prompts)
        
        self._save_step_checkpoint(2)

        # Step 3: Clean data (remove invalid responses before labeling)
        cleaned_df = self.clean_data(responses_df)
        
        self._save_step_checkpoint(3)

        # Step 4: Label data (dual-task labeling - only clean data)
        labeled_df = self.label_data(cleaned_df)
        
        self._save_step_checkpoint(4)

        # Step 4.5: Prepare jailbreak training data (NEW - V09)
        # Supplements with WildJailbreak if insufficient real jailbreak succeeded samples
        labeled_df_augmented = self.prepare_jailbreak_training_data(labeled_df)


        # Step 5: Prepare datasets for BOTH classifiers
        datasets = self.prepare_datasets(labeled_df_augmented)
        
        self._save_step_checkpoint(5)
        
        # Step 6: Train refusal classifier WITH cross-validation
        refusal_cv_results = self.train_refusal_classifier(
            datasets['refusal']['full_dataset']
        )

        self._save_step_checkpoint(6)
        
        # CRITICAL: Memory cleanup before training second classifier
        gc.collect()
        if DEVICE.type == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                if hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            gc.collect()
            print("🧹 MPS memory cleanup between classifiers complete")
        elif DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            print("🧹 CUDA memory cleanup between classifiers complete")

        # Step 7: Train jailbreak detector WITH cross-validation
        jailbreak_cv_results = self.train_jailbreak_detector(
            datasets['jailbreak']['full_dataset']
        )

        self._save_step_checkpoint(7)
        
        # CRITICAL: Memory cleanup before training second classifier
        gc.collect()
        if DEVICE.type == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                if hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            gc.collect()
            print("🧹 MPS memory cleanup between classifiers complete")
        elif DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            print("🧹 CUDA memory cleanup between classifiers complete")
        
        # Create test_df from CV results for analysis
        # CV returns predictions and labels - reconstruct test_df
        test_df = self._create_test_df_from_cv_results(
            refusal_cv_results,
            jailbreak_cv_results,
            labeled_df_augmented
        )

        # Step 8: Run adversarial testing
        adversarial_results = self.run_adversarial_testing(test_df)
        
        self._save_step_checkpoint(8)

        # Step 9: Run analyses (with adversarial results)
        analysis_results = self.run_analyses(test_df, adversarial_results)
        
        self._save_step_checkpoint(9)

        # Step 10: Generate visualizations
        refusal_history = refusal_cv_results.get('history', {})
        jailbreak_history = jailbreak_cv_results.get('history', {})
        figures = self.generate_visualizations(refusal_history, jailbreak_history, analysis_results)
        
        self._save_step_checkpoint(10)

        # Step 11: Generate reports
        self.generate_reports(refusal_cv_results, jailbreak_cv_results, analysis_results, figures)

        self._save_step_checkpoint(11)


        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE (DUAL CLASSIFIERS TRAINED)")
        print("="*60)

    def generate_prompts(self) -> Dict[str, List[str]]:
        """Step 1: Generate prompts."""
        print_banner("STEP 1: GENERATING PROMPTS", width=60)

        generator = PromptGenerator(self.api_keys['openai'])
        prompts = generator.generate_all_prompts()
        generator.save_prompts(prompts, data_raw_path, timestamp=self.run_timestamp)

        return prompts

    def collect_responses(self, prompts: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Step 2: Collect LLM responses.

        UPDATED (Phase 2/3): Uses parallel processing if enabled in config.
        """
        print_banner("STEP 2: COLLECTING RESPONSES", width=60)

        collector = ResponseCollector(
            self.api_keys['anthropic'],
            self.api_keys['openai'],
            self.api_keys.get('google', None)
        )

        # Use parallel processing if async is enabled
        responses_df = collector.collect_all_responses(
            prompts,
            parallel=API_CONFIG.get('use_async', True),
            resume_from_checkpoint=self.resume_from_checkpoint
        )

        # Save responses to disk
        timestamp = self.run_timestamp
        responses_path = os.path.join(data_responses_path, f"responses_{timestamp}.pkl")
        responses_df.to_pickle(responses_path)
        print(f"💾 Responses saved: {responses_path}")

        return responses_df

    def label_data(self, responses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Label responses using LLM Judge (dual-task).

        Labels ONLY clean data (after cleaning), saving API costs.

        UPDATED (Phase 2/3): Uses parallel processing if enabled in config.
        """
        print_banner("STEP 4: LABELING DATA WITH LLM JUDGE (DUAL-TASK)", width=60)

        # Initialize labeler with OpenAI API key (for GPT-4o judge)
        labeler = DataLabeler(api_key=self.api_keys['openai'])

        # Use checkpointed version if async is enabled
        if API_CONFIG.get('use_async', True):
            labeled_df = labeler.label_dataset_with_checkpoints(
                responses_df,
                resume_from_checkpoint=self.resume_from_checkpoint
            )
            # Rename jailbreak_success to jailbreak_label for consistency with rest of pipeline
            if 'jailbreak_success' in labeled_df.columns:
                labeled_df = labeled_df.rename(columns={'jailbreak_success': 'jailbreak_label'})
            return labeled_df
        else:
            # Sequential labeling (original implementation)
            refusal_labels = []
            is_jailbreak_attempts = []
            jailbreak_labels = []
            refusal_confidences = []
            jailbreak_confidences = []

            for idx, row in tqdm(responses_df.iterrows(), total=len(responses_df), desc="Dual-Task LLM Judge Labeling"):
                refusal_label, is_jailbreak_attempt, jailbreak_label, refusal_conf, jailbreak_conf = labeler.label_response(
                    response=row['response'],
                    prompt=row['prompt']
                )
                refusal_labels.append(refusal_label)
                is_jailbreak_attempts.append(is_jailbreak_attempt)
                jailbreak_labels.append(jailbreak_label)
                refusal_confidences.append(refusal_conf)
                jailbreak_confidences.append(jailbreak_conf)

            responses_df['refusal_label'] = refusal_labels
            responses_df['is_jailbreak_attempt'] = is_jailbreak_attempts
            responses_df['jailbreak_label'] = jailbreak_labels
            responses_df['refusal_confidence'] = refusal_confidences
            responses_df['jailbreak_confidence'] = jailbreak_confidences

        # Print refusal label distribution
        print(f"\n{'='*60}")
        print(f"REFUSAL LABELING SUMMARY")
        print(f"{'='*60}")
        for i in range(-1, 3):
            count = (responses_df['refusal_label'] == i).sum()
            pct = safe_divide(count, len(responses_df), default=0.0) * 100
            label_name = labeler.get_label_name(i)
            print(f"  {label_name}: {count} ({pct:.1f}%)")

        # Print jailbreak label distribution
        print(f"\n{'='*60}")
        print(f"JAILBREAK DETECTION SUMMARY")
        print(f"{'='*60}")
        for i in [0, 1]:
            count = (responses_df['jailbreak_label'] == i).sum()
            pct = safe_divide(count, len(responses_df), default=0.0) * 100
            label_name = labeler.get_jailbreak_label_name(i)
            print(f"  {label_name}: {count} ({pct:.1f}%)")

        # Print confidence statistics
        print(f"\n{'='*60}")
        print(f"CONFIDENCE STATISTICS")
        print(f"{'='*60}")
        valid_refusal = responses_df[responses_df['refusal_label'] != -1]
        valid_jailbreak = responses_df[responses_df['jailbreak_label'] != -1]

        # Use config threshold for low confidence
        low_conf_threshold = LABELING_CONFIG['low_confidence_threshold']

        if len(valid_refusal) > 0:
            avg_ref_conf = valid_refusal['refusal_confidence'].mean()
            low_conf_ref = (valid_refusal['refusal_confidence'] < low_conf_threshold).sum()
            print(f"  Refusal - Avg Confidence: {avg_ref_conf:.1f}%")
            print(f"  Refusal - Low confidence (<{low_conf_threshold}%): {low_conf_ref} ({low_conf_ref/len(valid_refusal)*100:.1f}%)")

        if len(valid_jailbreak) > 0:
            avg_jb_conf = valid_jailbreak['jailbreak_confidence'].mean()
            low_conf_jb = (valid_jailbreak['jailbreak_confidence'] < low_conf_threshold).sum()
            print(f"  Jailbreak - Avg Confidence: {avg_jb_conf:.1f}%")
            print(f"  Jailbreak - Low confidence (<{low_conf_threshold}%): {low_conf_jb} ({low_conf_jb/len(valid_jailbreak)*100:.1f}%)")

        # Analyze labeling quality
        print(f"\n{'='*60}")
        print("LABELING QUALITY ANALYSIS")
        print(f"{'='*60}")
        quality_analyzer = LabelingQualityAnalyzer(verbose=True)
        quality_results = quality_analyzer.analyze_full(responses_df)

        # Use run timestamp for consistency
        timestamp = self.run_timestamp

        # Save quality analysis results
        quality_analysis_path = os.path.join(quality_review_path, f"labeling_quality_analysis_{timestamp}.json")
        quality_analyzer.save_results(quality_results, quality_analysis_path)

        # Export low-confidence samples for review
        if quality_results['low_confidence']['low_both_count'] > 0:
            flagged_samples_path = os.path.join(quality_review_path, f"label_quality_flagged_samples_{timestamp}.csv")
            quality_analyzer.export_flagged_samples(responses_df, flagged_samples_path, threshold=LABELING_CONFIG['low_confidence_threshold'])

        # Save labeled data
        labeled_path = os.path.join(data_processed_path, f"labeled_responses_{timestamp}.pkl")
        responses_df.to_pickle(labeled_path)
        print(f"\n✓ Saved labeled data to {labeled_path}")

        return responses_df

    def clean_data(self, responses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Clean and validate data quality BEFORE labeling.

        Removes duplicates, outliers, and invalid data before expensive labeling.
        This saves API costs by not labeling garbage data.

        Args:
            responses_df: Raw responses DataFrame (unlabeled)

        Returns:
            Cleaned DataFrame ready for labeling
        """
        print("\n" + "="*60)
        print("STEP 3: CLEANING DATA (BEFORE LABELING)")
        print("="*60)

        # Initialize cleaner
        cleaner = DataCleaner()

        # Get outlier report first
        report = cleaner.get_outlier_report(responses_df)

        if report['issues_found']:
            print(f"\n📋 Outlier Report:")
            print(f"   Total samples: {report['total_samples']}")
            for issue in report['issues_found']:
                print(f"   • {issue['type']}: {issue['count']} ({issue['percentage']:.2f}%)")
            print(f"   Recommendation: {report['recommendation']}")
        else:
            print(f"\n✅ No data quality issues detected!")
            print(f"   Total samples: {report['total_samples']}")

        # Clean the data
        strategy = DATA_CLEANING_CONFIG['default_strategy']
        cleaned_df = cleaner.clean_dataset(responses_df, strategy=strategy)

        # Save cleaned data
        timestamp = self.run_timestamp
        cleaned_path = os.path.join(data_processed_path, f"cleaned_responses_{timestamp}.pkl")
        cleaned_df.to_pickle(cleaned_path)
        print(f"\n✓ Saved cleaned data to {cleaned_path}")

        return cleaned_df

    def prepare_datasets(self, labeled_df: pd.DataFrame) -> Dict:
        """
        Step 5: Prepare datasets for BOTH classifiers.

        Creates full datasets (for CV training) and test DataFrame (for analysis).
        Also saves train/val/test splits to disk for reference.

        Returns:
            Dictionary with structure:
            {
                'refusal': {full_dataset, test_df},
                'jailbreak': {full_dataset, test_df},
                'labeled_df': labeled_df (for CV training)
            }
        """
        print("\n" + "="*60)
        print("STEP 5: PREPARING DATASETS (DUAL CLASSIFIERS)")
        print("="*60)

        # Validate required columns exist before proceeding
        required_cols = ['prompt', 'response', 'refusal_label', 'jailbreak_label']
        missing_cols = [col for col in required_cols if col not in labeled_df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns for dataset preparation: {missing_cols}\n"
                f"This usually means labeling failed or was skipped. "
                f"Expected columns: {required_cols}"
            )

        # Filter out error labels (-1) from refusal labels
        error_count = (labeled_df['refusal_label'] == -1).sum()
        if error_count > 0:
            print(f"⚠️  Filtering out {error_count} error-labeled samples")
            labeled_df = labeled_df[labeled_df['refusal_label'] != -1].copy()
            print(f"✓ Remaining samples: {len(labeled_df)}")

        # Filter out NaN values in critical columns (COMPREHENSIVE FILTERING)
        # This prevents downstream errors in analyzers (PerModelAnalyzer, AttentionVisualizer, AdversarialTester)
        initial_count = len(labeled_df)
        labeled_df = labeled_df.dropna(subset=['response', 'prompt', 'model', 'refusal_label', 'jailbreak_label']).copy()
        nan_filtered = initial_count - len(labeled_df)

        if nan_filtered > 0:
            print(f"⚠️  Filtered out {nan_filtered} samples with NaN values in critical columns")
            print(f"✓ Remaining samples: {len(labeled_df)}")

        if len(labeled_df) == 0:
            raise ValueError("No valid samples remaining after filtering! Check data quality.")

        # Check if stratified split is possible
        # Stratification requires at least 2 samples per class in the smallest split
        label_counts = labeled_df['refusal_label'].value_counts()
        min_class_count = label_counts.min()
        
        # Calculate minimum samples needed for stratified split
        # Need at least 2 samples per class in the smallest split (test split)
        test_split_size = DATASET_CONFIG['test_split']
        val_split_size = DATASET_CONFIG['val_split']
        min_samples_needed = 2  # sklearn requires at least 2 samples per class
        
        # Determine if stratification is safe
        use_stratify = min_class_count >= min_samples_needed
        
        if not use_stratify:
            print(f"\n⚠️  WARNING: Small class counts detected!")
            print(f"   Minimum class count: {min_class_count}")
            print(f"   Class distribution: {dict(label_counts)}")
            print(f"   → Using NON-STRATIFIED split to avoid errors")
            print(f"   → This is expected for small test runs")
        
        # Split data (same splits for both classifiers to maintain consistency)
        train_df, temp_df = train_test_split(
            labeled_df,
            test_size=(1 - DATASET_CONFIG['train_split']),
            random_state=DATASET_CONFIG['random_seed'],
            stratify=labeled_df['refusal_label'] if use_stratify else None
        )

        # Use safe_divide from Utils for robust calculation
        val_size = safe_divide(
            DATASET_CONFIG['val_split'], 
            DATASET_CONFIG['val_split'] + DATASET_CONFIG['test_split'],
            default=0.5
        )
        
        # Check stratification for second split
        # After first split, check if temp_df has enough samples per class
        if use_stratify:
            temp_label_counts = temp_df['refusal_label'].value_counts()
            temp_min_count = temp_label_counts.min()
            use_stratify_second = temp_min_count >= min_samples_needed
            
            if not use_stratify_second:
                print(f"\n⚠️  WARNING: Cannot stratify val/test split!")
                print(f"   Temp split min class count: {temp_min_count}")
                print(f"   → Using NON-STRATIFIED split for val/test")
        else:
            use_stratify_second = False
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            random_state=DATASET_CONFIG['random_seed'],
            stratify=temp_df['refusal_label'] if use_stratify_second else None
        )

        print(f"\nSplit sizes:")
        print(f"  Train: {len(train_df)} ({len(train_df)/len(labeled_df)*100:.1f}%)")
        print(f"  Val: {len(val_df)} ({len(val_df)/len(labeled_df)*100:.1f}%)")
        print(f"  Test: {len(test_df)} ({len(test_df)/len(labeled_df)*100:.1f}%)")

        # Save splits
        timestamp = self.run_timestamp
        train_df.to_pickle(os.path.join(data_splits_path, f"train_{timestamp}.pkl"))
        val_df.to_pickle(os.path.join(data_splits_path, f"val_{timestamp}.pkl"))
        test_df.to_pickle(os.path.join(data_splits_path, f"test_{timestamp}.pkl"))

        # Add 'label' column for backward compatibility with analysis modules
        # Analysis modules expect 'label' column, so copy 'refusal_label' to 'label'
        train_df['label'] = train_df['refusal_label']
        val_df['label'] = val_df['refusal_label']
        test_df['label'] = test_df['refusal_label']

        # Initialize tokenizer (shared by both classifiers)
        print("\nInitializing tokenizer...")
        self.tokenizer = RobertaTokenizer.from_pretrained(MODEL_CONFIG['model_name'])

        # Prepare FULL datasets for cross-validation training
        print("\n--- Preparing Full Datasets for Cross-Validation ---")

        # Refusal classifier full dataset (uses all labeled data)
        refusal_full_dataset = ClassificationDataset(
            labeled_df['response'].tolist(),
            labeled_df['refusal_label'].tolist(),
            self.tokenizer
        )

        print(f"✓ Refusal classifier full dataset created:")
        print(f"  Total samples: {len(refusal_full_dataset):,}")

        # Jailbreak detector full dataset (uses all labeled data)
        jailbreak_full_dataset = ClassificationDataset(
            labeled_df['response'].tolist(),
            labeled_df['jailbreak_label'].tolist(),
            self.tokenizer
        )

        print(f"✓ Jailbreak detector full dataset created:")
        print(f"  Total samples: {len(jailbreak_full_dataset):,}")

        print(f"\n{'='*60}")
        print("NOTE: Cross-validation will handle train/val/test splitting internally")
        print(f"{'='*60}")

        return {
            'refusal': {
                'full_dataset': refusal_full_dataset,
                'test_df': test_df  # Will be populated by CV results
            },
            'jailbreak': {
                'full_dataset': jailbreak_full_dataset,
                'test_df': test_df  # Will be populated by CV results
            },
            'labeled_df': labeled_df  # Keep for reference
        }

    def train_refusal_classifier(self, full_dataset) -> Dict:
        """
        Step 6: Train RoBERTa refusal classifier WITH cross-validation.

        Uses K-fold cross-validation to:
        1. Evaluate model performance robustly
        2. Train final model on full train+val data
        3. Evaluate on held-out test set

        Args:
            full_dataset: Complete dataset (CV handles splitting internally)

        Returns:
            Dictionary with CV results and test performance
        """
        print("\n" + "="*60)
        print("STEP 6: TRAINING REFUSAL CLASSIFIER WITH CROSS-VALIDATION")
        print("="*60)

        # Use the standalone CV training function
        cv_results = train_with_cross_validation(
            full_dataset=full_dataset,
            model_class=RefusalClassifier,
            k_folds=CROSS_VALIDATION_CONFIG['default_folds'],
            test_split=DATASET_CONFIG['test_split'],
            class_names=CLASS_NAMES,
            save_final_model=True,
            final_model_path=os.path.join(
                models_path,
                f"{EXPERIMENT_CONFIG['experiment_name']}_refusal_best.pt"
            )
        )

        # Load the trained model into self.refusal_model for later use
        self.refusal_model = RefusalClassifier(num_classes=len(CLASS_NAMES))
        checkpoint = safe_load_checkpoint(cv_results['final_model_path'], DEVICE)
        self.refusal_model.load_state_dict(checkpoint['model_state_dict'])
        self.refusal_model = self.refusal_model.to(DEVICE)
        self.refusal_model.eval()

        print(f"\n✓ Refusal classifier training complete (CV-validated)")
        print(f"   CV Mean F1: {cv_results['cv_results']['overall']['f1_macro']['mean']:.4f} ± {cv_results['cv_results']['overall']['f1_macro']['std']:.4f}")
        print(f"   Test F1: {cv_results['test_results']['f1_macro']:.4f}")

        return cv_results

    def _get_viz_path(self, subdirectory: str, model_type: str, viz_name: str) -> str:
        """
        Generate standardized visualization path with SMART subdirectory routing.
        
        CRITICAL FIX: Routes core visualizations to main folder, specialized analysis to subfolders.
        
        Routing Rules:
        - Core visualizations (confusion_matrix, training_curves, etc.) → Main visualizations folder
        - Specialized analysis (attention, SHAP, error, etc.) → Respective subfolders
        - ALL files get classifier prefix: refusal_ or jailbreak_
        
        Args:
            subdirectory: Target subdirectory path (e.g., reporting_viz_path, error_analysis_path)
            model_type: Model identifier ('refusal', 'jailbreak', or 'combined')
            viz_name: Descriptive name of visualization
        
        Returns:
            Full path with standardized filename in appropriate directory
            
        Examples:
            >>> # Core visualization - stays in main folder
            >>> self._get_viz_path(reporting_viz_path, 'refusal', 'training_curves')
            '/path/to/Visualizations/refusal_training_curves_20251116.png'
            
            >>> # Specialized analysis - goes to subfolder
            >>> self._get_viz_path(error_analysis_path, 'refusal', 'error_distribution')
            '/path/to/Visualizations/Error Analysis/refusal_error_distribution_20251116.png'
        """
        filename = f"{model_type}_{viz_name}_{self.run_timestamp}.png"
        
        # Core visualizations that should stay in main folder (for PDF reports)
        # These are always called with reporting_viz_path as subdirectory
        CORE_VISUALIZATIONS = {
            'confusion_matrix', 'training_curves', 'class_distribution',
            'per_class_f1', 'per_model_f1', 'confidence_distributions',
            'adversarial_robustness', 'vulnerability_heatmap', 'vulnerability_comparison'
        }
        
        # If this is a core visualization, keep it in main visualizations folder
        # All specialized analysis already passes correct subdirectory (error_analysis_path, etc.)
        return os.path.join(subdirectory, filename)

    def _create_test_df_from_cv_results(self, refusal_cv_results: Dict, jailbreak_cv_results: Dict, labeled_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create test DataFrame from CV results for analysis.

        CV splits data internally, so we need to extract the test indices
        and reconstruct the test DataFrame with proper columns.
        
        Handles two scenarios:
        1. Models freshly trained: Use split_info['test_indices'] from CV
        2. Models loaded from checkpoint: Use full labeled_df as test set

        Args:
            refusal_cv_results: Refusal classifier CV results
            jailbreak_cv_results: Jailbreak detector CV results
            labeled_df: Original labeled DataFrame

        Returns:
            Test DataFrame ready for analysis
        """
        # Check if we have split_info (freshly trained) or loaded from checkpoint
        if 'split_info' in refusal_cv_results:
            # Scenario 1: Models were freshly trained - use test indices from CV
            test_indices = refusal_cv_results['split_info']['test_indices']
            test_df = labeled_df.iloc[test_indices].copy()
            print(f"\n✓ Created test DataFrame from CV split:")
        else:
            # Scenario 2: Models loaded from checkpoint - use full labeled_df as test set
            # This happens when starting from Step 8 (analyze-only)
            test_df = labeled_df.copy()
            print(f"\n✓ Using full labeled DataFrame as test set (models loaded from checkpoint):")
        
        # Add 'label' column for backward compatibility with analysis modules
        test_df['label'] = test_df['refusal_label']

        print(f"  Test samples: {len(test_df):,}")
        print(f"  Refusal class distribution:")
        for i in range(len(CLASS_NAMES)):
            count = (test_df['refusal_label'] == i).sum()
            pct = safe_divide(count, len(test_df), 0) * 100
            print(f"    {CLASS_NAMES[i]}: {count} ({pct:.1f}%)")

        return test_df

    def prepare_jailbreak_training_data(self, labeled_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare jailbreak training data with WildJailbreak supplementation if needed.

        NEW METHOD (V09): Supplements training data when modern LLMs successfully
        defend against all jailbreak attempts, resulting in insufficient positive samples.

        Args:
            labeled_df: DataFrame with labeled data (includes jailbreak_label column)

        Returns:
            DataFrame with jailbreak training data (real + WildJailbreak if needed)
        """
        print(f"\n\n")
        print(f"{'#'*60}")
        print(f"{'#'*60}")
        print(f"##  STEP 4.5: WILDJAILBREAK SUPPLEMENTATION")
        print(f"{'#'*60}")
        print(f"{'#'*60}")
        print(f"\n🔧 DEBUG INFO:")
        print(f"   Input dataframe shape: {labeled_df.shape}")
        print(f"   WILDJAILBREAK_CONFIG['enabled'] = {WILDJAILBREAK_CONFIG['enabled']}")
        print(f"   Jailbreak succeeded count: {(labeled_df['jailbreak_label'] == 1).sum()}")

        if not WILDJAILBREAK_CONFIG['enabled']:
            print(f"\n📊 WildJailbreak supplementation disabled")
            # Add data_source column for consistency
            labeled_df['data_source'] = 'real'
            return labeled_df

        print(f"\n{'='*60}")
        print(f"📊 PREPARING JAILBREAK TRAINING DATA")
        print(f"{'='*60}")

        # Calculate threshold from config (no hardcoded values)
        threshold = int(DATASET_CONFIG['total_prompts'] * (WILDJAILBREAK_CONFIG['min_threshold_percentage'] / 100))

        # Count real jailbreak succeeded samples (jailbreak_label == 1)
        real_jailbreak_succeeded = labeled_df[labeled_df['jailbreak_label'] == 1]
        real_count = len(real_jailbreak_succeeded)

        print(f"\n  Threshold: {threshold} samples ({WILDJAILBREAK_CONFIG['min_threshold_percentage']}% of {DATASET_CONFIG['total_prompts']} prompts)")
        print(f"  Real jailbreak succeeded samples: {real_count}")

        # Add data_source column to real data
        labeled_df_copy = labeled_df.copy()
        labeled_df_copy['data_source'] = 'real'

        # Check if supplementation needed
        if real_count >= threshold:
            print(f"  ✓ Sufficient real data - no supplementation needed")
            print(f"  Real data: 100%")
            print(f"{'='*60}\n")

            # Save labeled data with data_source column (even without WildJailbreak)
            # This ensures labeled_responses file always reflects the data used in Step 5
            timestamp = self.run_timestamp
            labeled_path = os.path.join(data_processed_path, f"labeled_responses_{timestamp}.pkl")
            labeled_df_copy.to_pickle(labeled_path)
            print(f"✓ Saved labeled data to {labeled_path}\n")

            return labeled_df_copy

        # Need supplementation
        samples_needed = threshold - real_count
        print(f"  ⚠️  Insufficient data - need {samples_needed} more samples")
        print(f"\n  Loading WildJailbreak dataset...")

        try:
            # Check if datasets library is installed
            try:
                print(f"  ✓ 'datasets' library found (version {datasets.__version__})")
            except ImportError:
                raise ImportError(
                    "\n\n"
                    "="*60 + "\n"
                    "❌ CRITICAL ERROR: 'datasets' library not installed!\n"
                    "="*60 + "\n"
                    "WildJailbreak supplementation requires the HuggingFace datasets library.\n"
                    "\n"
                    "Install with:\n"
                    "  pip install datasets\n"
                    "\n"
                    "Or install all requirements:\n"
                    "  pip install -r requirements.txt\n"
                    "="*60
                )

            # Initialize WildJailbreak loader
            loader = WildJailbreakLoader(random_seed=WILDJAILBREAK_CONFIG['random_seed'])

            # Load and sample
            wildjailbreak_samples = loader.load_and_sample(n_samples=samples_needed)

            if len(wildjailbreak_samples) == 0:
                raise RuntimeError(
                    f"WildJailbreak loader returned 0 samples! "
                    f"Requested {samples_needed} samples but got none. "
                    f"Cannot proceed with jailbreak training."
                )

            # Apply quality filters
            wildjailbreak_filtered = wildjailbreak_samples[
                (wildjailbreak_samples['prompt'].str.len() >= WILDJAILBREAK_CONFIG['min_prompt_length']) &
                (wildjailbreak_samples['prompt'].str.len() <= WILDJAILBREAK_CONFIG['max_prompt_length']) &
                (wildjailbreak_samples['response'].str.len() >= WILDJAILBREAK_CONFIG['min_response_length']) &
                (wildjailbreak_samples['response'].str.len() <= WILDJAILBREAK_CONFIG['max_response_length'])
            ].copy()

            print(f"  ✓ Loaded {len(wildjailbreak_filtered)} WildJailbreak samples (after quality filters)")

            # Combine datasets
            combined_df = pd.concat([labeled_df_copy, wildjailbreak_filtered], ignore_index=True)

            # Calculate composition using safe_divide from Utils
            total_jailbreak_succeeded = len(combined_df[combined_df['jailbreak_label'] == 1])
            real_percentage = safe_divide(real_count, total_jailbreak_succeeded, default=0.0) * 100
            wildjailbreak_percentage = 100 - real_percentage

            print(f"\n  {'='*56}")
            print(f"  📈 JAILBREAK TRAINING DATA COMPOSITION")
            print(f"  {'='*56}")
            print(f"  Real data:        {real_count:4d} samples ({real_percentage:5.1f}%)")
            print(f"  WildJailbreak:    {len(wildjailbreak_filtered):4d} samples ({wildjailbreak_percentage:5.1f}%)")
            print(f"  Total succeeded:  {total_jailbreak_succeeded:4d} samples")
            print(f"  {'='*56}")

            # Warning if too much supplementation
            if wildjailbreak_percentage > WILDJAILBREAK_CONFIG['warn_threshold']:
                print(f"\n  ⚠️  WARNING: {wildjailbreak_percentage:.1f}% of data from WildJailbreak")
                print(f"  Consider:")
                print(f"    1. Generating more aggressive jailbreak prompts")
                print(f"    2. Using adversarial prompt engineering techniques")
                print(f"    3. Lowering min_threshold_percentage (currently {WILDJAILBREAK_CONFIG['min_threshold_percentage']}%)")

            print(f"{'='*60}\n")

            
            # Save augmented labeled data (with WildJailbreak samples)
            timestamp = self.run_timestamp
            labeled_augmented_path = os.path.join(data_processed_path, f"labeled_responses_{timestamp}.pkl")
            combined_df.to_pickle(labeled_augmented_path)
            print(f"✓ Saved augmented labeled data to {labeled_augmented_path}\n")


            return combined_df

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ CRITICAL ERROR: WildJailbreak supplementation failed!")
            print(f"{'='*60}")
            print(f"\nError: {e}")
            print(f"\nThis means jailbreak training CANNOT proceed because:")
            print(f"  - Real jailbreak succeeded samples: {real_count}")
            print(f"  - Threshold required: {threshold}")
            print(f"  - Shortfall: {samples_needed} samples")
            print(f"\nFull traceback:")
            traceback.print_exc()
            print(f"\n{'='*60}")
            print(f"SOLUTION: Fix the WildJailbreak loading error above")
            print(f"{'='*60}\n")

            # Return original data - training will be skipped due to zero samples
            return labeled_df_copy

    def train_jailbreak_detector(self, full_dataset) -> Dict:
        """
        Step 7: Train RoBERTa jailbreak detector WITH cross-validation.

        Uses K-fold cross-validation to:
        1. Evaluate model performance robustly
        2. Train final model on full train+val data
        3. Evaluate on held-out test set

        Args:
            full_dataset: Complete dataset (CV handles splitting internally)

        Returns:
            Dictionary with CV results and test performance
        """
        print("\n" + "="*60)
        print("STEP 7: TRAINING JAILBREAK DETECTOR WITH CROSS-VALIDATION")
        print("="*60)

        # Check for class imbalance before training
        all_labels = full_dataset.labels
        class_counts = [all_labels.count(i) for i in range(len(JAILBREAK_CLASS_NAMES))]

        print(f"\nOverall class distribution:")
        for i, (class_name, count) in enumerate(zip(JAILBREAK_CLASS_NAMES, class_counts)):
            pct = safe_divide(count, len(all_labels), 0) * 100
            print(f"  {class_name}: {count:,} ({pct:.1f}%)")

        # Check for zero samples
        zero_classes = [i for i, count in enumerate(class_counts) if count == 0]
        if zero_classes:
            print(f"\n{'='*60}")
            print(f"🛑 JAILBREAK DETECTOR TRAINING SKIPPED")
            print(f"{'='*60}")
            print(f"\nREASON: Zero samples in class(es): {zero_classes}")
            for i, count in enumerate(class_counts):
                status = "❌ ZERO" if i in zero_classes else f"✓ {count:,}"
                print(f"  Class {i} ({JAILBREAK_CLASS_NAMES[i]}): {status}")

            print(f"\nWildJailbreak supplementation should have fixed this!")
            print(f"Check the output above for WildJailbreak loading errors.")
            print(f"{'='*60}\n")

            return {'status': 'skipped', 'reason': 'zero_samples', 'class_counts': class_counts}

        # Use the standalone CV training function
        cv_results = train_with_cross_validation(
            full_dataset=full_dataset,
            model_class=JailbreakDetector,
            k_folds=CROSS_VALIDATION_CONFIG['default_folds'],
            test_split=DATASET_CONFIG['test_split'],
            class_names=JAILBREAK_CLASS_NAMES,
            save_final_model=True,
            final_model_path=os.path.join(
                models_path,
                f"{EXPERIMENT_CONFIG['experiment_name']}_jailbreak_best.pt"
            )
        )

        # Load the trained model into self.jailbreak_model for later use
        self.jailbreak_model = JailbreakDetector(num_classes=len(JAILBREAK_CLASS_NAMES))
        checkpoint = safe_load_checkpoint(cv_results['final_model_path'], DEVICE)
        self.jailbreak_model.load_state_dict(checkpoint['model_state_dict'])
        self.jailbreak_model = self.jailbreak_model.to(DEVICE)
        self.jailbreak_model.eval()

        print(f"\n✓ Jailbreak detector training complete (CV-validated)")
        print(f"   CV Mean F1: {cv_results['cv_results']['overall']['f1_macro']['mean']:.4f} ± {cv_results['cv_results']['overall']['f1_macro']['std']:.4f}")
        print(f"   Test F1: {cv_results['test_results']['f1_macro']:.4f}")

        return cv_results



    def run_adversarial_testing(self, test_df: pd.DataFrame) -> Dict:
        """
        Step 8 (NEW): Run adversarial robustness testing via paraphrasing.
        
        This is now a SEPARATE STEP that:
        1. Checks if results already exist
        2. If yes, loads and returns them
        3. If no, runs paraphrasing and saves results
        
        Args:
            test_df: Test DataFrame
            
        Returns:
            Dictionary with adversarial testing results
        """
        print("\n" + "="*60)
        print("STEP 8: ADVERSARIAL TESTING (PARAPHRASING)")
        print("="*60)
        
        timestamp = self.run_timestamp
        output_path = os.path.join(analysis_results_path, f"adversarial_testing_{timestamp}.json")
        
# =============================================================================
#         # Check if results already exist
#         available = self.detect_available_data()
#         if 'adversarial_results' in available:
#             existing_path = available['adversarial_results']['path']
#             age_hours = available['adversarial_results']['age_hours']
#             
#             print("\n" + "="*60)
#             print("📂 EXISTING ADVERSARIAL RESULTS DETECTED")
#             print("="*60)
#             print(f"  File: {available['adversarial_results']['basename']}")
#             print(f"  Age: {age_hours:.1f} hours")
#             print("\n  Loading existing results instead of re-running paraphrasing...")
#             print("  (This saves time and API costs!)")
#             print("="*60 + "\n")
#             
#             # Load existing results
#             with open(existing_path, 'r') as f:
#                 adv_results = json.load(f)
#             
#             print("✅ Loaded existing adversarial testing results")
#             return adv_results
# =============================================================================
        
        # No existing results - run adversarial testing
        print("\n  No existing adversarial results found.")
        print("  Running paraphrasing tests (this may take a while)...\n")
        
        adversarial_tester = AdversarialTester(
            self.refusal_model, self.tokenizer, DEVICE, self.api_keys['openai']
        )
        adv_results = adversarial_tester.test_robustness(test_df)
        
        # Save results
        adversarial_tester.save_results(adv_results, output_path)
        
        print("\n✅ Adversarial testing complete")
        return adv_results


    def _load_adversarial_results(self) -> Optional[Dict]:
        """
        Load existing adversarial testing results.
        
        Returns:
            Dictionary with adversarial results or None if not found
        """
        available = self.detect_available_data()
        
        if 'adversarial_results' not in available:
            return None
        
        existing_path = available['adversarial_results']['path']
        
        print("\n📂 Loading adversarial results from previous run...")
        print(f"   File: {available['adversarial_results']['basename']}")
        
        try:
            with open(existing_path, 'r') as f:
                adv_results = json.load(f)
            print("   ✓ Loaded successfully")
            return adv_results
        except Exception as e:
            print(f"   ⚠️  Failed to load: {e}")
            return None


    def run_analyses(self, test_df: pd.DataFrame, adversarial_results: Dict = None) -> Dict:
        """
        Step 9 (MOVED from Step 8): Run all analyses for BOTH classifiers.
        
        REFACTORED:
        - Now accepts adversarial_results as a parameter instead of generating them
        - If adversarial_results is None, tries to load from disk
        - If still None, logs a warning but continues with other analyses
        
        Args:
            test_df: Test DataFrame
            adversarial_results: Optional pre-computed adversarial results
            
        Returns:
            Dictionary with all analysis results
        """
        print("\n" + "="*60)
        print("STEP 9: RUNNING ANALYSES (BOTH CLASSIFIERS)")
        print("="*60)

        # Validate models exist
        if self.refusal_model is None:
            raise RuntimeError("Refusal model not trained! Run train_refusal_classifier() first.")
        if self.jailbreak_model is None:
            raise RuntimeError("Jailbreak model not trained! Run train_jailbreak_detector() first.")

        analysis_results = {}

        # Use run timestamp for all analysis outputs
        timestamp = self.run_timestamp

        # ═══════════════════════════════════════════════════════
        # REFUSAL CLASSIFIER ANALYSIS
        # ═══════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("REFUSAL CLASSIFIER ANALYSIS")
        print("="*60)

        # STEP 0: Generate predictions from refusal model on test set
        print("\n--- Generating Refusal Predictions on Test Set ---")
        test_dataset = ClassificationDataset(
            test_df['response'].tolist(),
            test_df['refusal_label'].tolist(),
            self.tokenizer
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=TRAINING_CONFIG.get('batch_size', 16),
            shuffle=False,
            num_workers=0
        )

        all_preds = []
        all_labels = []
        all_probs = []

        self.refusal_model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating predictions"):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels_batch = batch['label']

                outputs = self.refusal_model(input_ids, attention_mask)
                # Handle both dict and tensor returns
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs

                probs = torch.softmax(logits, dim=1)
                preds_batch = torch.argmax(logits, dim=1)

                all_preds.extend(preds_batch.cpu().numpy())
                all_labels.extend(labels_batch.numpy())
                all_probs.extend(probs.cpu().numpy())

        # Convert to numpy arrays
        preds = np.array(all_preds)
        labels = np.array(all_labels)
        probs_array = np.array(all_probs)
        confidences = np.max(probs_array, axis=1)  # Max probability as confidence

        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        macro_precision = precision_score(labels, preds, average='macro', zero_division=0)
        macro_recall = recall_score(labels, preds, average='macro', zero_division=0)

        # Store in analysis_results
        analysis_results['predictions'] = {
            'preds': preds,
            'labels': labels,
            'confidences': confidences,
            'probabilities': probs_array,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall
        }

        print(f"✓ Generated predictions: Accuracy={accuracy:.4f}, F1={macro_f1:.4f}")

        # Per-model analysis
        print("\n--- Per-Model Analysis ---")
        per_model_analyzer = PerModelAnalyzer(self.refusal_model, self.tokenizer, DEVICE, class_names=CLASS_NAMES, task_type='refusal')
        per_model_results = per_model_analyzer.analyze(test_df)
        per_model_analyzer.save_results(
            per_model_results,
            os.path.join(analysis_results_path, f"refusal_per_model_analysis_{timestamp}.json")
        )
        analysis_results['per_model'] = per_model_results

        # Confidence analysis
        print("\n--- Confidence Analysis ---")
        confidence_analyzer = ConfidenceAnalyzer(self.refusal_model, self.tokenizer, DEVICE, class_names=CLASS_NAMES, task_type='refusal')
        conf_results = confidence_analyzer.analyze(test_df)
        confidence_analyzer.save_results(
            conf_results,
            os.path.join(analysis_results_path, f"refusal_confidence_analysis_{timestamp}.json")
        )
        analysis_results['confidence'] = conf_results
        
        
        # Adversarial Testing Results
        print("\n--- Adversarial Testing Results ---")

        if adversarial_results is not None:
            # Results provided as parameter (just generated in Step 8)
            print("  ✓ Using adversarial results from Step 8")
            analysis_results['adversarial'] = adversarial_results
        else:
            # Try to load from disk
            print("  Checking for existing adversarial results...")
            adversarial_results = self._load_adversarial_results()
            
            if adversarial_results is not None:
                print("  ✓ Loaded adversarial results from previous run")
                analysis_results['adversarial'] = adversarial_results
            else:
                print("  ⚠️  No adversarial results found")
                print("      Run Step 8 (Adversarial Testing) to generate them")
                analysis_results['adversarial'] = None
                
        
        # Attention visualization
        print("\n--- Attention Visualization ---")
        attention_viz = AttentionVisualizer(self.refusal_model, self.tokenizer, DEVICE, class_names=CLASS_NAMES, task_type='refusal')
        attention_results = attention_viz.analyze_samples(
        test_df,
        num_samples=INTERPRETABILITY_CONFIG['attention_samples_per_class'],
        output_dir=attention_analysis_path,
        timestamp=timestamp,
        classifier_name='refusal'
        )
        analysis_results['attention'] = attention_results
        
        
        # Attention visualization - JAILBREAK DETECTOR
        print("\n--- Attention Visualization (Jailbreak Detector) ---")
        jailbreak_attention_viz = AttentionVisualizer(
            self.jailbreak_model, 
            self.tokenizer, 
            DEVICE, 
            class_names=JAILBREAK_CLASS_NAMES, 
            task_type='jailbreak'
        )
        jailbreak_attention_results = jailbreak_attention_viz.analyze_samples(
        test_df,
        num_samples=INTERPRETABILITY_CONFIG['attention_samples_per_class'],
        output_dir=attention_analysis_path,
        timestamp=timestamp,
        classifier_name='jailbreak'
        )
        analysis_results['jailbreak_attention'] = jailbreak_attention_results
        

        # SHAP analysis (if enabled)
        if INTERPRETABILITY_CONFIG['shap_enabled']:
            print("\n--- SHAP Analysis ---")
            try:
                shap_analyzer = ShapAnalyzer(self.refusal_model, self.tokenizer, DEVICE, class_names=CLASS_NAMES)
                shap_results = shap_analyzer.analyze_samples(
                test_df,
                num_samples=INTERPRETABILITY_CONFIG['shap_samples'],
                output_dir=shap_analysis_path, 
                timestamp=timestamp,
                classifier_name='refusal'
                )
                analysis_results['shap'] = shap_results
            except ImportError:
                print("⚠️  SHAP not installed - skipping SHAP analysis")
                print("   Install with: pip install shap")
                analysis_results['shap'] = None
            except Exception as e:
                print(f"⚠️  SHAP analysis failed: {e}")
                analysis_results['shap'] = None
        else:
            print("\n--- SHAP Analysis (Disabled) ---")
            analysis_results['shap'] = None

        # Power Law Analysis (Refusal Classifier)
        print("\n--- Power Law Analysis (Refusal Classifier) ---")
        power_law_analyzer = PowerLawAnalyzer(
            self.refusal_model, self.tokenizer, DEVICE, 
            class_names=CLASS_NAMES,
            model_type="Refusal Classifier"
        )
        power_law_results = power_law_analyzer.analyze_all(
            test_df,
            np.array(preds),
            np.array(confidences),
            output_dir=power_law_viz_path,  # FIXED: Use correct subdirectory!
            timestamp=timestamp  # NEW: Add timestamp to filenames
        )
        analysis_results['power_law'] = power_law_results

        # ═══════════════════════════════════════════════════════
        # JAILBREAK DETECTOR ANALYSIS
        # ═══════════════════════════════════════════════════════
        #print("\n" + "="*60)
        #print("JAILBREAK DETECTOR ANALYSIS")
        #print("="*60)

        jailbreak_analyzer = JailbreakAnalysis(
            self.jailbreak_model,
            self.refusal_model,
            self.tokenizer,
            DEVICE
        )
        jailbreak_results = jailbreak_analyzer.analyze_full(test_df)
        jailbreak_analyzer.save_results(
            jailbreak_results,
            os.path.join(analysis_results_path, f"jailbreak_analysis_{timestamp}.json")
        )
        analysis_results['jailbreak'] = jailbreak_results

        # Power Law Analysis (Jailbreak Detector)
        print("\n--- Power Law Analysis (Jailbreak Detector) ---")
        jailbreak_power_law_analyzer = PowerLawAnalyzer(
            self.jailbreak_model, self.tokenizer, DEVICE, 
            class_names=JAILBREAK_CLASS_NAMES,
            model_type="Jailbreak Detector"
        )
        jailbreak_power_law_results = jailbreak_power_law_analyzer.analyze_all(
            test_df,
            jailbreak_results['predictions']['preds'],
            jailbreak_results['predictions']['confidences'],
            output_dir=power_law_viz_path,
            timestamp=timestamp  # NEW: Add timestamp to filenames
        )
        analysis_results['jailbreak_power_law'] = jailbreak_power_law_results

        # ═══════════════════════════════════════════════════════
        # REFUSAL-JAILBREAK CORRELATION ANALYSIS (Phase 2)
        # ═══════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("REFUSAL-JAILBREAK CORRELATION ANALYSIS")
        print("="*60)

        correlation_analyzer = RefusalJailbreakCorrelationAnalyzer(
            refusal_preds=preds,
            jailbreak_preds=jailbreak_results['predictions']['preds'],
            refusal_labels=labels,
            jailbreak_labels=jailbreak_results['predictions']['labels'],
            texts=test_df['response'].tolist(),
            refusal_class_names=CLASS_NAMES,
            jailbreak_class_names=JAILBREAK_CLASS_NAMES
        )
        correlation_results = correlation_analyzer.run_full_analysis(
                output_dir=correlation_viz_path, 
                timestamp=timestamp,
                classifier_name='correlation'  # or 'refusal_jailbreak'
                )
        correlation_analyzer.save_analysis_results(
                timestamp=timestamp,
                classifier_name='correlation'
                )
        
        analysis_results['correlation'] = correlation_results

        # ═══════════════════════════════════════════════════════
        # ERROR ANALYSIS (Phase 2)
        # ═══════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("ERROR ANALYSIS: REFUSAL CLASSIFIER")
        print("="*60)

        # Create test dataset for error analysis
        refusal_test_dataset = ClassificationDataset(
            test_df['response'].tolist(),
            test_df['refusal_label'].tolist(),
            self.tokenizer
        )

        # Run comprehensive error analysis (7 modules)
        refusal_error_results = run_error_analysis(
            model=self.refusal_model,
            dataset=refusal_test_dataset,
            tokenizer=self.tokenizer,
            device=DEVICE,
            class_names=CLASS_NAMES,
            task_type='refusal',
            output_dir=error_analysis_path,
            timestamp=timestamp
        )
        analysis_results['error_analysis_refusal'] = refusal_error_results

        # Error analysis for jailbreak detector
        print("\n" + "="*60)
        print("ERROR ANALYSIS: JAILBREAK DETECTOR")
        print("="*60)

        jailbreak_test_dataset = ClassificationDataset(
            test_df['response'].tolist(),
            test_df['jailbreak_label'].tolist(),
            self.tokenizer
        )

        jailbreak_error_results = run_error_analysis(
            model=self.jailbreak_model,
            dataset=jailbreak_test_dataset,
            tokenizer=self.tokenizer,
            device=DEVICE,
            class_names=JAILBREAK_CLASS_NAMES,
            task_type='jailbreak',
            output_dir=error_analysis_path,
            timestamp=timestamp
        )
        analysis_results['error_analysis_jailbreak'] = jailbreak_error_results

        return analysis_results

    def generate_visualizations(self, refusal_history: Dict, jailbreak_history: Dict, analysis_results: Dict):
        """
        Step 9: Generate all visualizations for both classifiers.
        
        ALL visualizations are now organized into subdirectories with standardized naming:
        - Format: {model_type}_{viz_name}_{timestamp}.png
        - Subdirectories: Main visualization folder for core metrics, specialized folders for advanced analysis

        Returns:
            Dictionary of matplotlib figure objects for use in report generation
        """
        print("\n" + "="*60)
        print("STEP 9: GENERATING VISUALIZATIONS")
        print("="*60)

        # Use run timestamp for all visualizations
        timestamp = self.run_timestamp
        
        visualizer = Visualizer()
        jailbreak_visualizer = Visualizer(class_names=JAILBREAK_CLASS_NAMES)
        figures = {}

        # ═══════════════════════════════════════════════════════════════════
        # CORE TRAINING VISUALIZATIONS (Main Visualizations Folder)
        # ═══════════════════════════════════════════════════════════════════
        
        print("\n📊 Generating core training visualizations...")
        
        # Training curves - Refusal Classifier
        figures['refusal_training_curves'] = visualizer.plot_training_curves(
            refusal_history,
            self._get_viz_path(reporting_viz_path, 'refusal', 'training_curves')
        )

        # Training curves - Jailbreak Detector
        figures['jailbreak_training_curves'] = jailbreak_visualizer.plot_training_curves(
            jailbreak_history,
            self._get_viz_path(reporting_viz_path, 'jailbreak', 'training_curves')
        )

        # Get predictions and labels for refusal classifier
        preds = analysis_results['predictions']['preds']
        labels = analysis_results['predictions']['labels']

        # Confusion matrix - Refusal Classifier
        cm = confusion_matrix(labels, preds)
        figures['confusion_matrix'] = visualizer.plot_confusion_matrix(
            cm,
            self._get_viz_path(reporting_viz_path, 'refusal', 'confusion_matrix')
        )

        # Class distribution - Refusal Classifier
        figures['class_distribution'] = visualizer.plot_class_distribution(
            labels,
            self._get_viz_path(reporting_viz_path, 'refusal', 'class_distribution')
        )

        # Per-class F1 - Refusal Classifier
        per_class_f1 = {}
        if 'models' in analysis_results['per_model']:
            for model_name, model_data in analysis_results['per_model']['models'].items():
                if 'f1_per_class' in model_data:
                    for cls, f1 in model_data['f1_per_class'].items():
                        if cls not in per_class_f1:
                            per_class_f1[cls] = []
                        per_class_f1[cls].append(f1)

        avg_per_class_f1 = {cls: np.mean(scores) for cls, scores in per_class_f1.items()}
        visualizer.plot_per_class_f1(
            avg_per_class_f1,
            self._get_viz_path(reporting_viz_path, 'refusal', 'per_class_f1')
        )

        # Per-model F1 - Shows performance across LLM models (Claude, GPT, Gemini)
        visualizer.plot_per_model_f1(
            analysis_results['per_model']['models'],  # FIX: Access 'models' sub-dict
            self._get_viz_path(reporting_viz_path, 'refusal', 'per_model_f1')
        )

        # Adversarial robustness - Refusal Classifier (only if results exist)
        if analysis_results.get('adversarial') is not None:
            visualizer.plot_adversarial_robustness(
                analysis_results['adversarial'],
                self._get_viz_path(reporting_viz_path, 'refusal', 'adversarial_robustness')
            )
        else:
            print("⚠️  Skipping adversarial robustness plot - no results available")

        # Confidence distributions - Refusal Classifier
        confidences = analysis_results['predictions']['confidences']
        visualizer.plot_confidence_distributions(
            labels,
            confidences,
            self._get_viz_path(reporting_viz_path, 'refusal', 'confidence_distributions')
        )
        
        print("  ✓ Core visualizations saved to:", reporting_viz_path)

        # ═══════════════════════════════════════════════════════════════════
        # CORRELATION ANALYSIS (Correlation Analysis Subfolder)
        # ═══════════════════════════════════════════════════════════════════
        
        # Note: Correlation analysis is already generated in run_analyses() at line 1494
        # It generates its own visualizations with proper subdirectory and naming
        print("\n📈 Correlation analysis visualizations:")
        print("  ✓ Already generated in Step 8 (run_analyses)")
        print(f"  📁 Location: {correlation_viz_path}")

        # ═══════════════════════════════════════════════════════════════════
        # ERROR ANALYSIS (Error Analysis Subfolder)
        # ═══════════════════════════════════════════════════════════════════
        
        # Note: Error analysis is already generated in run_analyses() at lines 1519, 1541
        # It generates its own visualizations with proper subdirectory and naming
        print("\n🔍 Error analysis visualizations:")
        print("  ✓ Already generated in Step 8 (run_analyses)")
        print(f"  📁 Location: {error_analysis_path}")

        # ═══════════════════════════════════════════════════════════════════
        # ATTENTION ANALYSIS (Attention Analysis Subfolder)
        # ═══════════════════════════════════════════════════════════════════
        
        # Note: Attention analysis is already generated in run_analyses() at line 1393
        # It generates its own visualizations with proper subdirectory and naming
        print("\n🧠 Attention analysis visualizations:")
        print("  ✓ Already generated in Step 8 (run_analyses)")
        print(f"  📁 Location: {attention_analysis_path}")

        # ═══════════════════════════════════════════════════════════════════
        # SHAP ANALYSIS (SHAP Analysis Subfolder)
        # ═══════════════════════════════════════════════════════════════════
        
        # Note: SHAP analysis is already generated in run_analyses() at line 1405
        # It generates its own visualizations with proper subdirectory and naming
        print("\n🔬 SHAP analysis visualizations:")
        if analysis_results.get('shap') is not None:
            print("  ✓ Already generated in Step 8 (run_analyses)")
            print(f"  📁 Location: {shap_analysis_path}")
        else:
            print("  ⚠️  SHAP analysis was skipped (disabled or failed)")
            print(f"  📁 Expected location: {shap_analysis_path}")

        # ═══════════════════════════════════════════════════════════════════
        # POWER LAW ANALYSIS (Power Law Analysis Subfolder)
        # ═══════════════════════════════════════════════════════════════════
        
        # Note: Power law analysis is already generated in run_analyses() at lines 1429, 1461
        # It generates its own visualizations with proper subdirectory and naming
        print("\n📐 Power law analysis visualizations:")
        print("  ✓ Already generated in Step 8 (run_analyses)")
        print(f"  📁 Location: {power_law_viz_path}")

        print("\n" + "="*60)
        print("✅ ALL VISUALIZATIONS ORGANIZED INTO SUBDIRECTORIES")
        print("="*60)
        print(f"📁 Main visualizations: {visualizations_path}")
        print(f"📁 Attention analysis: {attention_analysis_path}")
        print(f"📁 SHAP analysis: {shap_analysis_path}")
        print(f"📁 Correlation analysis: {correlation_viz_path}")
        print(f"📁 Error analysis: {error_analysis_path}")
        print(f"📁 Power law analysis: {power_law_viz_path}")
        print("="*60)
        
        return figures

    def generate_reports(self, refusal_history: Dict, jailbreak_history: Dict, analysis_results: Dict, figures: Dict = None):
        """
        Step 10: Generate comprehensive PDF reports for both classifiers.

        Args:
            refusal_history: Training history for refusal classifier
            jailbreak_history: Training history for jailbreak detector
            analysis_results: Results from run_analyses()
            figures: Dictionary of matplotlib figure objects from generate_visualizations()
        """
        print("\n" + "="*60)
        print("STEP 10: GENERATING REPORTS")
        print("="*60)

        # Check if reportlab is available
        if not REPORTLAB_AVAILABLE:
            print("\n⚠️  PDF report generation disabled - reportlab not installed")
            print("   Install with: pip install reportlab")
            print("   Skipping report generation...")
            return

        timestamp = self.run_timestamp

        # If figures not provided (e.g., starting from step 10), we can't generate reports
        if figures is None:
            print("\n⚠️  Cannot generate reports: No visualization figures available")
            print("   Please run from Step 9 to generate visualizations first")
            return

        # Initialize report generator for refusal classifier
        refusal_report_gen = ReportGenerator(class_names=CLASS_NAMES)

        # Initialize report generator for jailbreak detector
        jailbreak_report_gen = ReportGenerator(class_names=JAILBREAK_CLASS_NAMES)

        # ═══════════════════════════════════════════════════════
        # REFUSAL CLASSIFIER REPORTS
        # ═══════════════════════════════════════════════════════

        print("\n--- Generating Refusal Classifier Report ---")

        # Calculate metrics from predictions
        preds = np.array(analysis_results['predictions']['preds'])
        labels = np.array(analysis_results['predictions']['labels'])
        
        # Filter out error labels
        valid_mask = labels != -1
        valid_preds = preds[valid_mask]
        valid_labels = labels[valid_mask]
        
        refusal_metrics = {
            'accuracy': float(accuracy_score(valid_labels, valid_preds)),
            'macro_f1': float(f1_score(valid_labels, valid_preds, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(valid_labels, valid_preds, average='weighted', zero_division=0)),
            'macro_precision': float(precision_score(valid_labels, valid_preds, average='macro', zero_division=0)),
            'macro_recall': float(recall_score(valid_labels, valid_preds, average='macro', zero_division=0))
        }

        # Extract metrics from analysis results
# =============================================================================
#         refusal_metrics = {
#             'accuracy': analysis_results['predictions']['accuracy'],
#             'macro_f1': analysis_results['predictions']['macro_f1'],
#             'weighted_f1': analysis_results['predictions']['weighted_f1'],
#             'macro_precision': analysis_results['predictions']['macro_precision'],
#             'macro_recall': analysis_results['predictions']['macro_recall']
#         }
# =============================================================================

        # Generate refusal classifier performance report
# =============================================================================
#         refusal_report_path = os.path.join(reports_path, f"refusal_performance_report_{timestamp}.pdf")
#         refusal_report_gen.generate_model_performance_report(
#             model_name="Refusal Classifier",
#             metrics=refusal_metrics,
#             confusion_matrix_fig=figures['confusion_matrix'],
#             training_curves_fig=figures['refusal_training_curves'],
#             class_distribution_fig=figures['class_distribution'],
#             output_path=refusal_report_path,
#             run_timestamp=timestamp  # NEW: Pass run timestamp to report
#         )
# =============================================================================


        # Build comprehensive metrics from analysis_results (no import needed - it's already loaded)
        # refusal_metrics = build_refusal_metrics_from_results(analysis_results)        
        
        refusal_data = analysis_results.copy()

        if 'correlation' in analysis_results:
            refusal_data['correlation'] = analysis_results['correlation']
        
        if 'adversarial' in analysis_results:
            refusal_data['adversarial'] = analysis_results['adversarial']
        
        refusal_metrics = build_refusal_metrics_from_results(refusal_data)

        
        # FIX: Read actual split sizes from metadata (saved by ExperimentRunner)
        metadata = analysis_results.get('metadata', {})
        
        train_size = metadata.get('num_train_samples', 'N/A')
        val_size = metadata.get('num_val_samples', 'N/A')
        
        # Fallback estimation if metadata unavailable
        if train_size == 'N/A' or val_size == 'N/A':
            test_size = len(analysis_results['predictions']['labels'])
            train_split = DATASET_CONFIG['train_split']
            val_split = DATASET_CONFIG['val_split']
            train_size = int(test_size * (train_split / DATASET_CONFIG['test_split']))
            val_size = int(test_size * (val_split / DATASET_CONFIG['test_split']))
        
        refusal_metrics['train_samples'] = train_size
        refusal_metrics['val_samples'] = val_size
        
        test_samples_count = len(analysis_results.get('predictions', {}).get('labels', []))
        refusal_metrics['test_samples'] = test_samples_count
        
        # Generate refusal classifier performance report
        refusal_report_path = os.path.join(reports_path, f"refusal_performance_report_{timestamp}.pdf")
        refusal_report_gen.generate_model_performance_report(
            model_name="Refusal Classifier",
            metrics=refusal_metrics,
            confusion_matrix_fig=figures['confusion_matrix'],
            training_curves_fig=figures['refusal_training_curves'],
            class_distribution_fig=figures['class_distribution'],
            output_path=refusal_report_path,
            run_timestamp=timestamp
        )

        print(f"   ✓ Saved: {os.path.basename(refusal_report_path)}")
        print(f"      Metrics: Accuracy={refusal_metrics['accuracy']:.3f}, F1={refusal_metrics['macro_f1']:.3f}")

        # Close refusal figures to free memory
        plt.close(figures['confusion_matrix'])
        plt.close(figures['refusal_training_curves'])
        plt.close(figures['class_distribution'])

        # ═══════════════════════════════════════════════════════
        # JAILBREAK DETECTOR REPORTS
        # ═══════════════════════════════════════════════════════

        print("\n--- Generating Jailbreak Detector Report ---")

        # Extract jailbreak analysis results
        print("\n--- Generating Jailbreak Detector Report ---")
        
        # Extract jailbreak analysis results
        if 'jailbreak' in analysis_results and analysis_results['jailbreak']:
            jailbreak_data = analysis_results['jailbreak'].copy()  # Make a copy to avoid modifying original
            
            # CRITICAL FIX: Add correlation and adversarial data from root level
            # These are stored at analysis_results level, not inside jailbreak dict
            if 'correlation' in analysis_results:
                jailbreak_data['correlation'] = analysis_results['correlation']
            
            if 'adversarial' in analysis_results:
                jailbreak_data['adversarial'] = analysis_results['adversarial']
            
            if 'metadata' in analysis_results:
                jailbreak_data['metadata'] = analysis_results['metadata']

            # Build comprehensive metrics using existing builder function (SAME AS REFUSAL!)
            jailbreak_metrics = build_jailbreak_metrics_from_results(jailbreak_data)
            
            
            # FIX: Read actual split sizes from metadata (saved by ExperimentRunner)
            metadata = analysis_results.get('metadata', {})
            
            train_size = metadata.get('num_train_samples', 'N/A')
            val_size = metadata.get('num_val_samples', 'N/A')
            
            if train_size == 'N/A' or val_size == 'N/A':
                test_size = len(analysis_results['predictions']['labels'])
                train_split = DATASET_CONFIG['train_split']
                val_split = DATASET_CONFIG['val_split']
                train_size = int(test_size * (train_split / DATASET_CONFIG['test_split']))
                val_size = int(test_size * (val_split / DATASET_CONFIG['test_split']))
            
            jailbreak_metrics['train_samples'] = train_size
            jailbreak_metrics['val_samples'] = val_size
            
            
            test_samples_count = len(jailbreak_data.get('predictions', {}).get('labels', []))
            jailbreak_metrics['test_samples'] = test_samples_count
            
            # Calculate total WildJailbreak from saved labeled_df
            try:
                labeled_files = glob.glob(os.path.join(data_processed_path, "labeled_responses_*.pkl"))
                if labeled_files:
                    labeled_df = pd.read_pickle(sorted(labeled_files)[-1])
                    total_wj = len(labeled_df[labeled_df['data_source'] == 'wildjailbreak'])
                    jailbreak_metrics['total_wildjailbreak_samples'] = total_wj
            except Exception:
                pass
            
            
            # Get predictions for visualizations
            if 'predictions' in jailbreak_data:
                jb_preds = jailbreak_data['predictions'].get('preds', [])
                jb_labels = jailbreak_data['predictions'].get('labels', [])
                
                if len(jb_preds) > 0 and len(jb_labels) > 0:
                    # Convert to numpy arrays
                    jb_preds = np.array(jb_preds)
                    jb_labels = np.array(jb_labels)
                    
                    # 1. CREATE CONFUSION MATRIX FIGURE (not in figures dict!)
                    jb_cm = confusion_matrix(jb_labels, jb_preds, labels=[0, 1])
                    jailbreak_cm_fig = plt.figure(figsize=(8, 6))
                    ax_cm = jailbreak_cm_fig.add_subplot(111)
                    
                    sns.heatmap(jb_cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=JAILBREAK_CLASS_NAMES,
                               yticklabels=JAILBREAK_CLASS_NAMES,
                               ax=ax_cm, cbar_kws={'label': 'Count'})
                    ax_cm.set_xlabel('Predicted Label', fontsize=12)
                    ax_cm.set_ylabel('True Label', fontsize=12)
                    ax_cm.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    
                    # 2. CREATE CLASS DISTRIBUTION FIGURE
                    jailbreak_class_dist_fig = plt.figure(figsize=(8, 6))
                    ax_dist = jailbreak_class_dist_fig.add_subplot(111)
                    
                    class_counts = [np.sum(jb_labels == 0), np.sum(jb_labels == 1)]
                    bars = ax_dist.bar(JAILBREAK_CLASS_NAMES, class_counts, color=JAILBREAK_PLOT_COLORS_LIST)
                    ax_dist.set_ylabel('Number of Samples', fontsize=12)
                    ax_dist.set_title('Jailbreak Class Distribution', fontsize=14, fontweight='bold')
                    ax_dist.grid(axis='y', alpha=0.3)
                    
                    # Add count labels on bars
                    for bar, count in zip(bars, class_counts):
                        height = bar.get_height()
                        ax_dist.text(bar.get_x() + bar.get_width()/2., height,
                                    str(count), ha='center', va='bottom', fontsize=11)
                    
                    plt.tight_layout()
                    
                    # 3. GET TRAINING CURVES (already in figures dict from generate_visualizations)
                    jailbreak_training_fig = figures.get('jailbreak_training_curves', None)
                    
                    # Generate comprehensive jailbreak performance report (MIRRORS REFUSAL REPORT!)
                    jailbreak_report_path = os.path.join(reports_path, f"jailbreak_performance_report_{timestamp}.pdf")
                    jailbreak_report_gen.generate_model_performance_report(
                        model_name="Jailbreak Detector",
                        metrics=jailbreak_metrics,
                        confusion_matrix_fig=jailbreak_cm_fig,
                        training_curves_fig=jailbreak_training_fig,
                        class_distribution_fig=jailbreak_class_dist_fig,
                        output_path=jailbreak_report_path,
                        run_timestamp=timestamp
                    )
                    
                    print(f"   ✓ Generated: {jailbreak_report_path}")
                    print(f"      Accuracy: {jailbreak_metrics['accuracy']*100:.1f}%, F1: {jailbreak_metrics['macro_f1']:.3f}")
                    
                    # Close created figures
                    plt.close(jailbreak_cm_fig)
                    plt.close(jailbreak_class_dist_fig)
                else:
                    print("   ⚠️  No jailbreak predictions available for report")
            else:
                print("   ⚠️  No jailbreak predictions in analysis results")
        else:
            print("   ⚠️  No jailbreak analysis results available")

        # Close jailbreak figure
        if 'jailbreak_training_curves' in figures:
            plt.close(figures['jailbreak_training_curves'])
        if 'jailbreak_confusion_matrix' in figures:
            plt.close(figures['jailbreak_confusion_matrix'])

        # ═══════════════════════════════════════════════════════
        # COMBINED EXECUTIVE SUMMARY
        # ═══════════════════════════════════════════════════════

        print("\n--- Executive Summary ---")
        print(f"   ℹ️  Executive summary generation not yet implemented")
        print(f"      (Would combine both classifiers into single report)")

        print("\n✓ Reports generated")
        print("   📁 Reports saved to:", reports_path)


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: Refusal Pipeline Module
Created on October 28, 2025
@author: ramyalsaffar
"""
