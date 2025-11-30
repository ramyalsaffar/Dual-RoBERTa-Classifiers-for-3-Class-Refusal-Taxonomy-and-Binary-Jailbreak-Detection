# ExperimentRunner Module
#------------------------
# Manages different experiment execution modes.
# Adapted from Alignment Tax project.
# All imports are in 01-Imports.py
###############################################################################


class ExperimentRunner:
    """Manages different experiment execution modes."""

    def __init__(self):
        self.pipeline = None
        self.api_keys = None

    def _get_split_size(self, split_name: str) -> int:
        """
        Get the size of a data split (train/val/test) from saved files.
        
        Args:
            split_name: 'train', 'val', or 'test'
            
        Returns:
            Number of samples in the split, or 'N/A' if file not found
        """
        try:
            # Find the most recent split file
            pattern = os.path.join(data_splits_path, f"{split_name}_*.pkl")
            files = glob.glob(pattern)
            
            if not files:
                return 'N/A'
            
            # Get most recent file
            latest_file = max(files, key=os.path.getmtime)
            
            # Load and get length
            df = pd.read_pickle(latest_file)
            return len(df)
        except Exception as e:
            print(f"⚠️  Could not load {split_name} split size: {e}")
            return 'N/A'

    def _read_api_keys_from_file(self) -> Dict[str, str]:
        """
        Read API keys from local file.

        File format (each line):
            OpenAI API Key: sk-...
            Anthropic API Key: sk-ant-...
            Google API Key: AI...

        Returns:
            Dictionary with keys: 'openai', 'anthropic', 'google'
            Returns None if file not found or parsing fails
        """
        try:
            if not os.path.exists(api_keys_file_path):
                print(f"⚠️  API keys file not found: {api_keys_file_path}")
                return None

            api_keys = {}
            with open(api_keys_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    # Strip ALL whitespace and remove non-printable characters
                    line = line.strip()
                    # Remove any BOM or hidden characters
                    line = ''.join(char for char in line if char.isprintable())

                    if not line:
                        continue

                    # Parse line: "OpenAI API Key: sk-..."
                    if ':' in line:
                        key_name, key_value = line.split(':', 1)
                        # Aggressively strip all whitespace
                        key_value = key_value.strip()

                        if 'OpenAI' in key_name:
                            api_keys['openai'] = key_value
                        elif 'Anthropic' in key_name:
                            api_keys['anthropic'] = key_value
                        elif 'Google' in key_name:
                            api_keys['google'] = key_value

            # Validate all keys are present
            # Required keys = all models in response_models config
            required_keys = []
            if 'claude' in API_CONFIG['response_models']:
                required_keys.append('anthropic')
            if 'gpt5' in API_CONFIG['response_models']:
                required_keys.append('openai')
            if 'gemini' in API_CONFIG['response_models']:
                required_keys.append('google')
            
            if all(k in api_keys for k in required_keys):
                return api_keys
            else:
                missing = [k for k in required_keys if k not in api_keys]
                print(f"❌ Missing required API keys: {', '.join(missing)}")
                return None

        except Exception as e:
            print(f"⚠️  Error reading API keys file: {e}")
            return None

    def _get_api_keys(self) -> Dict[str, str]:
        """
        Get API keys based on environment.

        Priority order:
        1. AWS Secrets Manager (if in AWS)
        2. Local API Keys file (if exists)
        3. Environment variables (.env)
        4. Manual input (last resort)

        Returns:
            Dictionary with keys: 'openai', 'anthropic', 'google'
        """
        if IS_AWS and AWS_AVAILABLE:
            # Use AWS Secrets Manager
            print("🔐 Retrieving API keys from AWS Secrets Manager...")
            try:
                secrets_handler = SecretsHandler(region=AWS_CONFIG['region'])

                return {
                    'openai': secrets_handler.get_api_key(AWS_CONFIG['secrets']['openai']),
                    'anthropic': secrets_handler.get_api_key(AWS_CONFIG['secrets']['anthropic']),
                    'google': secrets_handler.get_api_key(AWS_CONFIG['secrets']['google'])
                }
            except Exception as e:
                print(f"❌ Error retrieving secrets from AWS: {e}")
                print("Falling back to local methods...")

        # Local mode: Try API keys file first
        print("🔑 API Key Configuration")
        print("-" * 40)

        # Try reading from local API keys file
        file_keys = self._read_api_keys_from_file()
        if file_keys:
            print("✓ Loaded API keys from local file")
            print(f"  - OpenAI: {'*' * 20}{file_keys['openai'][-4:]} (length: {len(file_keys['openai'])})")
            print(f"  - Anthropic: {'*' * 20}{file_keys['anthropic'][-4:]} (length: {len(file_keys['anthropic'])})")
            print(f"  - Google: {'*' * 20}{file_keys['google'][-4:]} (length: {len(file_keys['google'])})")

            # Debug: Check for whitespace issues
            if file_keys['openai'].strip() != file_keys['openai']:
                print("  ⚠️  WARNING: OpenAI key has leading/trailing whitespace!")
            if file_keys['anthropic'].strip() != file_keys['anthropic']:
                print("  ⚠️  WARNING: Anthropic key has leading/trailing whitespace!")
            if file_keys['google'].strip() != file_keys['google']:
                print("  ⚠️  WARNING: Google key has leading/trailing whitespace!")

            return file_keys

        # Try loading from .env file
        try:
            load_dotenv()
            print("✓ Checking .env file...")
        except (ImportError, FileNotFoundError):
            pass  # dotenv is optional

        # Get or prompt for OpenAI key
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print("\nOpenAI API Key not found in file or .env")
            openai_key = getpass.getpass("Enter OpenAI API Key: ")

        # Get or prompt for Anthropic key
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if not anthropic_key:
            print("\nAnthropic API Key not found in file or .env")
            anthropic_key = getpass.getpass("Enter Anthropic API Key: ")

        # Get or prompt for Google key
        google_key = os.getenv('GOOGLE_API_KEY')
        if not google_key:
            print("\nGoogle API Key not found in file or .env")
            google_key = getpass.getpass("Enter Google API Key: ")

        print("\n✓ API keys configured")

        return {
            'openai': openai_key,
            'anthropic': anthropic_key,
            'google': google_key
        }

    def _check_and_prompt_for_resume(self) -> tuple:
        """
        Check for existing checkpoints/data and prompt user for execution strategy.

        Returns:
            Tuple of (resume_from_checkpoint: bool, start_step: int)
            - resume_from_checkpoint: Whether to resume from checkpoints (for Steps 2 & 4)
            - start_step: Which step to start from (1-9, where 1 = start from beginning)
        """
        # Check for checkpoints
        checkpoint_patterns = [
            'checkpoint_prompt_generation_*.pkl',
            'checkpoint_response_collection_*.pkl',
            'checkpoint_labeling_*.pkl',
            'checkpoint_wildjailbreak_loading_*.pkl'
        ]

        found_checkpoints = {}
        for pattern in checkpoint_patterns:
            checkpoints = glob.glob(os.path.join(data_checkpoints_path, pattern))
            if checkpoints:
                # Get most recent
                latest = max(checkpoints, key=os.path.getmtime)
                operation = pattern.split('_')[1]  # Extract operation name
                timestamp = os.path.basename(latest).split('_')[-1].replace('.pkl', '')
                age_hours = (time.time() - os.path.getmtime(latest)) / 3600

                found_checkpoints[operation] = {
                    'path': latest,
                    'timestamp': timestamp,
                    'age_hours': age_hours
                }

        # Check for intermediate data files
        temp_pipeline = RefusalPipeline(api_keys={'openai': '', 'anthropic': '', 'google': ''})
        available_data = temp_pipeline.detect_available_data()

        # If no checkpoints or data, start fresh
        if not found_checkpoints and not available_data:
            return (False, 1)

        # Display what's available
        print("\n" + "="*70)
        print("📋 EXISTING PIPELINE DATA DETECTED")
        print("="*70)

        # Show checkpoints
        if found_checkpoints:
            print("\n✓ Available Checkpoints (for in-progress operations):")
            for operation, info in found_checkpoints.items():
                print(f"  • {operation.title().replace('_', ' ')}: {info['age_hours']:.1f} hours old")

        # Show completed step data
        if available_data:
            print("\n✓ Completed Step Data:")
            step_names = {
                'responses': 'Step 2: Response Collection',
                'cleaned': 'Step 3: Data Cleaning',
                'labeled': 'Step 4: Data Labeling',
                'splits': 'Step 5: Dataset Preparation'
            }
            for key, data_info in available_data.items():
                step_name = step_names.get(key, key)
                print(f"  • {step_name}: {data_info['basename']} ({data_info['age_hours']:.1f} hours old)")

        print("\n" + "="*70)
        print("\n🎯 EXECUTION OPTIONS:")
        print("─"*70)
        print("1. Resume from latest checkpoints (continue in-progress operations)")
        print("   → Resumes Step 2 (responses) or Step 4 (labeling) if interrupted")
        print("\n2. Start from specific step (skip completed steps)")
        print("   → Load completed data and start from a later step")
        print("\n3. Start fresh (delete all checkpoints and ignore saved data)")
        print("   → Start pipeline from Step 1, delete all existing checkpoints")
        print("─"*70)

        # Get user choice
        while True:
            choice = input("\nSelect option (1/2/3): ").strip()

            if choice == '1':
                # Resume from checkpoints - detect which step to resume from
                print("\n✓ Analyzing available data to determine resume point...")
                
                # Determine the latest completed step with PROPER prerequisite checking
                latest_step = 1
                
                # Helper: Check if both models exist (required for Steps 8+)
                has_both_models = ('refusal_model' in available_data and 'jailbreak_model' in available_data)
                has_refusal_model = 'refusal_model' in available_data
                has_jailbreak_model = 'jailbreak_model' in available_data
                
                # CRITICAL: Use proper prerequisite checking (not just file existence)
                # Step 10 requires: Step 9 complete (visualizations)
                if 'reports' in available_data:
                    latest_step = 10
                
                # Step 9 requires: Step 8 complete (analysis) + both models
                elif 'visualizations' in available_data and has_both_models:
                    latest_step = 9
                
                # Step 8 complete: Adversarial testing results exist
                elif 'adversarial_results' in available_data and has_both_models:
                    latest_step = 8
                
                # Step 7 complete: Jailbreak model exists
                elif has_jailbreak_model:
                    latest_step = 7
                
                # Step 6 complete: Refusal model exists (but not jailbreak)
                elif has_refusal_model:
                    latest_step = 6
                
                # Step 5 complete: Data splits exist
                elif 'splits' in available_data:
                    latest_step = 5
                
                # Step 4 complete: Labeled data exists
                elif 'labeled' in available_data:
                    latest_step = 4
                
                # Step 3 complete: Cleaned data exists
                elif 'cleaned' in available_data:
                    latest_step = 3
                
                # Step 2 complete: Responses exist
                elif 'responses' in available_data:
                    latest_step = 2
                
                # ═════════════════════════════════════════════════════════════
                # CRITICAL FIX: SMART CHECKPOINT DETECTION
                # ═════════════════════════════════════════════════════════════
                # Checkpoints should ONLY be used if they represent LATER work
                # than completed steps. Don't resume Step 4 checkpoint if models
                # are already trained!
                # ═════════════════════════════════════════════════════════════
                
                checkpoint_step = None
                checkpoint_name = None
                
                # Determine which checkpoint (if any) to use
                # Check in reverse order (newest steps first) to resume from furthest point
                if 'wildjailbreak' in found_checkpoints and latest_step < 5:
                    # Only use wildjailbreak checkpoint if we haven't completed Step 5 yet
                    checkpoint_step = 5
                    checkpoint_name = "wildjailbreak"
                elif 'labeling' in found_checkpoints and latest_step < 4:
                    # Only use labeling checkpoint if we haven't completed Step 4 yet
                    checkpoint_step = 4
                    checkpoint_name = "labeling"
                elif 'response' in found_checkpoints and latest_step < 2:
                    # Only use response checkpoint if we haven't completed Step 2 yet
                    checkpoint_step = 2
                    checkpoint_name = "response"
                elif 'prompt' in found_checkpoints and latest_step < 1:
                    # Only use prompt checkpoint if we haven't completed Step 1 yet
                    checkpoint_step = 1
                    checkpoint_name = "prompt generation"
                
                # Use checkpoint ONLY if it's for later work than completed steps
                if checkpoint_step is not None:
                    print(f"✓ Found {checkpoint_name} checkpoint - will resume Step {checkpoint_step}")
                    return (True, checkpoint_step)
                else:
                    # No valid checkpoint OR checkpoint is for earlier work than completed steps
                    start_step = latest_step + 1 if latest_step < 10 else 1
                    
                    if found_checkpoints:
                        print(f"✓ Found old checkpoints but ignoring (later steps already complete)")
                    else:
                        print(f"✓ No active checkpoints found")
                    
                    print(f"✓ Latest completed step: Step {latest_step}")
                    print(f"✓ Will start from: Step {start_step}")
                    return (False, start_step)

            elif choice == '2':
                # Start from specific step
                print("\n" + "─"*70)
                print("📍 SELECT STARTING STEP:")
                print("─"*70)
                
                # Check model status for warnings
                has_both_models = ('refusal_model' in available_data and 'jailbreak_model' in available_data)
                has_refusal = 'refusal_model' in available_data
                has_jailbreak = 'jailbreak_model' in available_data

                # Build step list with availability indicators
                # Build step list with availability indicators
                steps = [
                    ("1", "Generate Prompts", 'prompts' in available_data),
                    ("2", "Collect Responses", 'responses' in available_data),
                    ("3", "Clean Data", 'cleaned' in available_data),
                    ("4", "Label Data", 'labeled' in available_data),
                    ("5", "Prepare Datasets", 'splits' in available_data),
                    ("6", "Train Refusal Classifier", has_refusal),
                    ("7", "Train Jailbreak Detector", has_jailbreak),
                    ("8", "Adversarial Testing", 'adversarial_results' in available_data),  # NEW STEP!
                    ("9", "Run Analyses", 'analysis_results' in available_data),
                    ("10", "Generate Visualizations", 'visualizations' in available_data),  # MOVED from 9 to 10
                    ("11", "Generate Reports", 'reports' in available_data)  # MOVED from 10 to 11
                ]

                for num, name, has_data in steps:
                    indicator = "✓" if has_data else " "
                    print(f"  {num}. {name:30} [{indicator}]")

                print("\n  Legend: [✓] = Can skip (data available)")
                print("─"*70)

                while True:
                    step_input = input("\nStart from step (1-11): ").strip()
                    try:
                        start_step = int(step_input)
                        if 1 <= start_step <= 11:
                            # CRITICAL: Warn if user is about to retrain existing models
                            if start_step == 6 and has_refusal and has_jailbreak:
                                print("\n⚠️  WARNING: Trained models already exist!")
                                print(f"   Refusal model: {available_data['refusal_model']['basename']}")
                                print(f"   Jailbreak model: {available_data['jailbreak_model']['basename']}")
                                print("\n   Starting from Step 6 will LOAD existing models (not retrain).")
                                print("   This is the RECOMMENDED behavior to save time.")
                                confirm = input("\n   Continue with Step 6? (y/n): ").strip().lower()
                                if confirm not in ['y', 'yes']:
                                    continue
                            
                            elif start_step == 7 and has_jailbreak:
                                print("\n⚠️  WARNING: Jailbreak model already exists!")
                                print(f"   Model: {available_data['jailbreak_model']['basename']}")
                                print("\n   Starting from Step 7 will LOAD existing model (not retrain).")
                                print("   This is the RECOMMENDED behavior to save time.")
                                confirm = input("\n   Continue with Step 7? (y/n): ").strip().lower()
                                if confirm not in ['y', 'yes']:
                                    continue
                            
                            # Validate that required data exists
                            try:
                                if start_step >= 3:
                                    temp_pipeline.load_data_for_step(start_step)
                                print(f"\n✓ Will start from Step {start_step}: {steps[start_step-1][1]}")
                                return (False, start_step)
                            except FileNotFoundError as e:
                                print(f"\n❌ Cannot start from Step {start_step}: {e}")
                                print("   Please select an earlier step or choose option 3 to start fresh.")
                                continue
                        else:
                            print("⚠️  Please enter a number between 1-11")
                    except ValueError:
                        print("⚠️  Please enter a valid number")

            elif choice == '3':
                # Start fresh
                print("\n⚠️  This will DELETE all checkpoints. Continue? (y/n): ", end='')
                confirm = input().strip().lower()
                if confirm in ['y', 'yes']:
                    print("\n✓ Starting fresh (deleting checkpoints...)")
                    # Cleanup all checkpoints
                    for operation in ['response_collection', 'labeling', 'wildjailbreak_loading']:
                        checkpoint_manager = CheckpointManager(
                            checkpoint_dir=data_checkpoints_path,
                            operation_name=operation,
                            auto_cleanup=False
                        )
                        deleted = checkpoint_manager.delete_all_checkpoints(confirm=True)
                    return (False, 1)
                else:
                    print("   Cancelled. Please select another option.")
                    continue

            else:
                print("⚠️  Please enter 1, 2, or 3")

    def _print_experiment_header(self, mode_name: str, description: str = ""):
        """Print formatted experiment header."""
        print("\n" + "="*60)
        print(f"{mode_name.upper()}")
        print("="*60)
        if description:
            print(description)
            print("="*60)

    def quick_test(self):
        """
        Run quick test with reduced samples.

        WHY: Allows rapid prototyping and testing without running full pipeline.
        Samples a subset of data at each stage to validate end-to-end workflow.
        """
        self._print_experiment_header(
            "Quick Test Mode",
            f"Testing with {EXPERIMENT_CONFIG['test_sample_size']} total prompts (distributed across categories)"
        )

        # Check for existing checkpoints and prompt user
        resume_from_checkpoint, start_step = self._check_and_prompt_for_resume()

        # Get API keys
        api_keys = self._get_api_keys()

        # Initialize pipeline with quick test mode and resume flag
        self.pipeline = RefusalPipeline(api_keys, resume_from_checkpoint=resume_from_checkpoint)

        # Override pipeline behavior for quick test
        print("\n🚀 Quick test mode active - using reduced dataset")
        print(f"   Test sample size: {EXPERIMENT_CONFIG['test_sample_size']}")
        print(f"   Reduced total prompts: {EXPERIMENT_CONFIG['test_sample_size']}")

        # Temporarily override dataset config for quick testing
        # WHY: Reduce total prompts to speed up data collection and testing
        original_total_prompts = DATASET_CONFIG['total_prompts']
        DATASET_CONFIG['total_prompts'] = EXPERIMENT_CONFIG['test_sample_size']

        try:
            # Prevent Mac from sleeping during execution
            with KeepAwake():
                if start_step == 1:
                    self.pipeline.run_full_pipeline()
                else:
                    self.pipeline.run_partial_pipeline(start_step=start_step)
        finally:
            # Restore original config
            # WHY: Prevent config pollution affecting other modes
            DATASET_CONFIG['total_prompts'] = original_total_prompts

    def full_experiment(self):
        """Run full experiment as configured."""
        self._print_experiment_header(
            "Full Experiment Mode",
            f"Running complete pipeline with {DATASET_CONFIG['total_prompts']} prompts"
        )

        # Check for existing checkpoints and prompt user
        resume_from_checkpoint, start_step = self._check_and_prompt_for_resume()

        # Get API keys
        api_keys = self._get_api_keys()

        # Run pipeline with sleep prevention and resume flag
        self.pipeline = RefusalPipeline(api_keys, resume_from_checkpoint=resume_from_checkpoint)
        with KeepAwake():
            if start_step == 1:
                self.pipeline.run_full_pipeline()
            else:
                self.pipeline.run_partial_pipeline(start_step=start_step)

    def train_only(self):
        """Train only (assumes data already collected)."""
        self._print_experiment_header(
            "Train Only Mode",
            "Loading existing data and training BOTH classifiers"
        )

        # Check if data exists
        labeled_data_path = os.path.join(data_processed_path, "labeled_responses.pkl")
        if not os.path.exists(labeled_data_path):
            print(f"❌ Error: Labeled data not found at {labeled_data_path}")
            print("Please run data collection first or use --full mode")
            return

        # Load data
        print(f"\nLoading data from {labeled_data_path}...")
        labeled_df = pd.read_pickle(labeled_data_path)
        print(f"✓ Loaded {len(labeled_df)} labeled samples")

        # Get API keys (use already-loaded keys from self.api_keys)
        api_keys = self.api_keys if hasattr(self, 'api_keys') and self.api_keys else {
            'openai': os.getenv('OPENAI_API_KEY') or getpass.getpass("OpenAI API Key (for analysis): ")
        }

        # Initialize pipeline
        self.pipeline = RefusalPipeline(api_keys)

        # Prepare datasets (returns dict with 'refusal' and 'jailbreak' keys)
        datasets = self.pipeline.prepare_datasets(labeled_df)

        # Train refusal classifier
        refusal_history = self.pipeline.train_refusal_classifier(
            datasets['refusal']['train_loader'],
            datasets['refusal']['val_loader']
        )

        # Train jailbreak detector
        jailbreak_history = self.pipeline.train_jailbreak_detector(
            datasets['jailbreak']['train_loader'],
            datasets['jailbreak']['val_loader']
        )

        # Run analyses (uses test_df from refusal datasets - same for both)
        analysis_results = self.pipeline.run_analyses(datasets['refusal']['test_df'])

        # Generate visualizations (using both histories)
        self.pipeline.generate_visualizations(refusal_history, jailbreak_history, analysis_results)

        print("\n✅ Training and analysis complete (BOTH classifiers trained)")


    def analyze_only(self, refusal_model_path: str = None, jailbreak_model_path: str = None, test_data_path: str = None):
        """
        Analysis only (load existing models for BOTH classifiers).

        Args:
            refusal_model_path: Path to trained refusal classifier (default: models/{experiment_name}_refusal_best.pt)
            jailbreak_model_path: Path to trained jailbreak detector (default: models/{experiment_name}_jailbreak_best.pt)
            test_data_path: Path to test data (default: data/splits/test.pkl)

        WHY: Allows rerunning analysis with different test sets or on different model checkpoints
        without retraining. Useful for validating model on new data or comparing model versions.
        """
        self._print_experiment_header(
            "Analysis Only Mode",
            "Loading trained models and running analysis (BOTH classifiers)"
        )

        # Determine model paths
        if refusal_model_path is None:
            refusal_model_path = os.path.join(models_path, f"{EXPERIMENT_CONFIG['experiment_name']}_refusal_best.pt")
        if jailbreak_model_path is None:
            jailbreak_model_path = os.path.join(models_path, f"{EXPERIMENT_CONFIG['experiment_name']}_jailbreak_best.pt")

        # Check if models exist
        if not os.path.exists(refusal_model_path):
            print(f"❌ Error: Refusal model not found at {refusal_model_path}")
            print("Please train models first or provide correct paths")
            return
        if not os.path.exists(jailbreak_model_path):
            print(f"❌ Error: Jailbreak model not found at {jailbreak_model_path}")
            print("Please train models first or provide correct paths")
            return

        # Determine test data path
        if test_data_path is None:
            test_data_path = os.path.join(data_splits_path, "test.pkl")

        # Check if test data exists
        if not os.path.exists(test_data_path):
            print(f"❌ Error: Test data not found at {test_data_path}")
            print("Please run full pipeline first or provide valid test data path")
            return

        # Load test data
        print(f"\nLoading test data from {test_data_path}...")
        test_df = pd.read_pickle(test_data_path)
        print(f"✓ Loaded {len(test_df)} test samples")
        
    
        # Calculate expected test sizes from CONFIG (no hardcoded numbers!)
        # Calculate expected sizes INCLUDING BUFFER
        
        num_models = len(API_CONFIG['response_models'])  # Dynamic model count
        
        test_split = DATASET_CONFIG['test_split']  # 0.15
        buffer_pct = EXPERIMENT_CONFIG['prompt_buffer_percentage'] / 100  # 0.20
        
        # Quick test: 150 + 20% = 180 prompts
        quick_prompts_with_buffer = int(EXPERIMENT_CONFIG['test_sample_size'] * (1 + buffer_pct))
        quick_total_responses = quick_prompts_with_buffer * num_models
        quick_test_expected = int(quick_total_responses * test_split)
        
        # Full experiment: 2000 + 20% = 2400 prompts
        full_prompts_with_buffer = int(DATASET_CONFIG['total_prompts'] * (1 + buffer_pct))
        full_total_responses = full_prompts_with_buffer * num_models
        full_test_expected = int(full_total_responses * test_split)
        
        # Tolerance: ±20%
        quick_tolerance = int(quick_test_expected * 0.20)
        full_tolerance = int(full_test_expected * 0.20)
        
        is_close_to_quick = abs(len(test_df) - quick_test_expected) < quick_tolerance
        is_close_to_full = abs(len(test_df) - full_test_expected) < full_tolerance

        if not is_close_to_quick and not is_close_to_full:
            print(f"\n⚠️  WARNING: Unexpected test set size!")
            print(f"   Test samples: {len(test_df)}")
            print(f"   Expected for quick test: ~{quick_test_expected} samples (±{quick_tolerance})")
            print(f"   Expected for full experiment: ~{full_test_expected} samples (±{full_tolerance})")
            
            response = input("   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("\n❌ Analysis cancelled. Please verify test data file.")
                return

        # Add 'label' column for refusal classifier analysis (points to refusal_label)
        # WHY: Saved test.pkl doesn't have 'label' column, but analysis modules expect it
        test_df['label'] = test_df['refusal_label']
        
        # Check if model column exists
        if 'model' not in test_df.columns:
            print("⚠️  WARNING: 'model' column missing - per-model analysis will be skipped")
            print("   To enable per-model analysis, re-run from Step 1 to include model column in test data")

        
        # Validate test data has required columns
        # WHY: Ensure test data contains both refusal and jailbreak labels for dual-task analysis
        required_columns = ['response', 'refusal_label', 'is_jailbreak_attempt', 'jailbreak_label']
        missing_columns = [col for col in required_columns if col not in test_df.columns]
        if missing_columns:
            print(f"❌ Error: Test data missing required columns: {missing_columns}")
            print(f"   Required: {required_columns}")
            print(f"   Found: {list(test_df.columns)}")
            return

        # Initialize tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_CONFIG['model_name'])

        # Extract timestamp from model filename (format: dual_RoBERTa_classifier_YYYYMMDD_HHMM_refusal_best.pt)
        # WHY: Use model's timestamp for all visualizations to maintain proper tracking and association
        model_timestamp = None
        try:
            # Extract from refusal model filename (preferred)
            refusal_basename = os.path.basename(refusal_model_path)
            # Pattern: dual_RoBERTa_classifier_20251115_1623_refusal_best.pt
            # Extract: 20251115_1623
            parts = refusal_basename.split('_')
            if len(parts) >= 4:
                # Find timestamp pattern (YYYYMMDD_HHMM)
                for i in range(len(parts) - 1):
                    if len(parts[i]) == 8 and parts[i].isdigit() and len(parts[i+1]) == 4 and parts[i+1].isdigit():
                        model_timestamp = f"{parts[i]}_{parts[i+1]}"
                        break
            print(f"✓ Extracted model timestamp: {model_timestamp}")
        except Exception as e:
            print(f"⚠️  Could not extract timestamp from model filename: {e}")
            print(f"   Using current timestamp instead")
            model_timestamp = get_timestamp('file')

        # Load refusal classifier (using safe_load_checkpoint for MPS bug workaround)
        print(f"\nLoading refusal classifier from {refusal_model_path}...")
        refusal_model = RefusalClassifier(num_classes=MODEL_CONFIG['num_classes'])
        refusal_checkpoint = safe_load_checkpoint(refusal_model_path, DEVICE)
        refusal_model.load_state_dict(refusal_checkpoint['model_state_dict'])
        refusal_model = refusal_model.to(DEVICE)
        refusal_model.eval()  # WHY: Set to evaluation mode (disables dropout, sets BatchNorm to eval)
        print(f"✓ Refusal classifier loaded (Best Val F1: {refusal_checkpoint['best_val_f1']:.4f})")
        
        refusal_cv_results = refusal_checkpoint.get('cv_results', {})
        
        # Extract training history for training curves
        refusal_history = refusal_checkpoint.get('history', {})
        if refusal_history:
            print(f"✓ Extracted refusal training history ({len(refusal_history.get('train_loss', []))} epochs)")
        else:
            print(f"⚠️  No training history found in refusal checkpoint")

        # Load jailbreak detector (using safe_load_checkpoint for MPS bug workaround)
        print(f"\nLoading jailbreak detector from {jailbreak_model_path}...")
        jailbreak_model = JailbreakDetector(num_classes=JAILBREAK_CONFIG['num_classes'])
        jailbreak_checkpoint = safe_load_checkpoint(jailbreak_model_path, DEVICE)
        jailbreak_model.load_state_dict(jailbreak_checkpoint['model_state_dict'])
        jailbreak_model = jailbreak_model.to(DEVICE)
        jailbreak_model.eval()  # WHY: Set to evaluation mode (disables dropout, sets BatchNorm to eval)
        print(f"✓ Jailbreak detector loaded (Best Val F1: {jailbreak_checkpoint['best_val_f1']:.4f})")

        # Run analyses
        print("\n" + "="*60)
        print("RUNNING ANALYSES (BOTH CLASSIFIERS)")
        print("="*60)

        analysis_results = {}

        # ═══════════════════════════════════════════════════════
        # REFUSAL CLASSIFIER ANALYSIS
        # ═══════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("REFUSAL CLASSIFIER ANALYSIS")
        print("="*60)

        # Per-model analysis
        print("\n--- Per-Model Analysis ---")
        per_model_analyzer = PerModelAnalyzer(refusal_model, tokenizer, DEVICE, class_names=CLASS_NAMES, task_type='refusal')
        per_model_results = per_model_analyzer.analyze(test_df)
        
        #per_model_analyzer.save_results(per_model_results, os.path.join(analysis_results_path, f"refusal_per_model_analysis_{model_timestamp}.json") # Already saved in RefusalPipeline

        
        analysis_results['per_model'] = per_model_results

        # Confidence analysis
        print("\n--- Confidence Analysis ---")
        confidence_analyzer = ConfidenceAnalyzer(refusal_model, tokenizer, DEVICE, class_names=CLASS_NAMES, task_type='refusal')
        conf_results = confidence_analyzer.analyze(test_df)
        
        # confidence_analyzer.save_results(conf_results, os.path.join(analysis_results_path, f"refusal_confidence_analysis_{model_timestamp}.json") # Already saved in RefusalPipeline

        analysis_results['confidence'] = conf_results
        
        # Extract predictions for visualization and metrics
        # Need to re-run evaluation to get preds, labels, confidences
        dataset = ClassificationDataset(
            texts=test_df['response'].tolist(),
            labels=test_df['refusal_label' if 'refusal_label' in test_df.columns else 'label'].tolist(),
            tokenizer=tokenizer,
            task_type='refusal',
            validate_labels=False
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Get predictions
        refusal_model.eval()
        preds, labels, confidences = [], [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                batch_labels = batch['label']
                
                logits = refusal_model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                batch_preds = torch.argmax(probs, dim=1).cpu().numpy()
                batch_confidences = probs.max(dim=1)[0].cpu().numpy()
                
                preds.extend(batch_preds)
                labels.extend(batch_labels.numpy())
                confidences.extend(batch_confidences.tolist())
        
        analysis_results['predictions'] = {
            'preds': preds,
            'labels': labels,
            'confidences': confidences
        }

        # ═══════════════════════════════════════════════════════════
        # NEW: CALCULATE COMPLETE PER-CLASS METRICS FOR REPORT
        # ═══════════════════════════════════════════════════════════
        print("\n--- Calculating Per-Class Metrics for Report ---")
        
        # Filter out error labels (-1)
        valid_mask = np.array(labels) != -1
        valid_labels = np.array(labels)[valid_mask]
        valid_preds = np.array(preds)[valid_mask]
        
        if len(valid_labels) > 0:
            
            report = classification_report(
                valid_labels,
                valid_preds,
                labels=list(range(len(CLASS_NAMES))),
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0
            )
            
            # Store per-class metrics in analysis_results
            analysis_results['per_class_metrics'] = {}
            for i, class_name in enumerate(CLASS_NAMES):
                if class_name in report:
                    analysis_results['per_class_metrics'][f'class_{i}'] = report[class_name]
            
            # Store metadata for report generation
            # Extract train/val sizes from CV results
            split_info = refusal_cv_results.get('split_info', {})
            train_val_size = split_info.get('train_val_size', 'N/A')
            test_size = split_info.get('test_size', 'N/A')
            
            # For K-fold CV: Calculate average train/val from fold ratio
            # K-fold uses (K-1)/K for train, 1/K for val per fold
            k_folds = CROSS_VALIDATION_CONFIG.get('default_folds', 5)
            if isinstance(train_val_size, int) and train_val_size > 0:
                # Average fold train size: (K-1)/K of train_val_size
                avg_train_size = int(train_val_size * (k_folds - 1) / k_folds)
                avg_val_size = train_val_size - avg_train_size
            else:
                avg_train_size = 'N/A'
                avg_val_size = 'N/A'
            
            # Store metadata for report generation
            analysis_results['metadata'] = {
                'test_df': test_df,
                'predictions': valid_preds.tolist(),
                'true_labels': valid_labels.tolist(),
                'confidences': np.array(confidences)[valid_mask].tolist(),
                'refusal_class_names': CLASS_NAMES,
                'cv_results': refusal_cv_results,
                'train_samples': avg_train_size,
                'val_samples': avg_val_size,
                'test_samples': test_size,
                'epochs': TRAINING_CONFIG.get('epochs', 3),
                'batch_size': TRAINING_CONFIG.get('batch_size', 16),
                'learning_rate': TRAINING_CONFIG.get('learning_rate', 2e-5),
                'warmup_steps': TRAINING_CONFIG.get('warmup_steps', 100),
                'weight_decay': TRAINING_CONFIG.get('weight_decay', 0.01),
                'gradient_clip': TRAINING_CONFIG.get('gradient_clip', 1.0),

            }
            
            print(f"✓ Calculated per-class metrics for {len(valid_preds)} samples")
            print(f"  Accuracy: {(valid_preds == valid_labels).mean():.4f}")
            
            # Print per-class support for verification
            for i, class_name in enumerate(CLASS_NAMES):
                if class_name in report:
                    support = int(report[class_name]['support'])
                    print(f"  {class_name} Support: {support}")
        else:
            print("⚠️  No valid samples for per-class metrics calculation")
            analysis_results['per_class_metrics'] = {}
            
            # Extract train/val sizes from CV results (same as if block)
            split_info = refusal_cv_results.get('split_info', {})
            train_val_size = split_info.get('train_val_size', 'N/A')
            test_size = split_info.get('test_size', 'N/A')
            
            k_folds = CROSS_VALIDATION_CONFIG.get('default_folds', 5)
            if isinstance(train_val_size, int) and train_val_size > 0:
                avg_train_size = int(train_val_size * (k_folds - 1) / k_folds)
                avg_val_size = train_val_size - avg_train_size
            else:
                avg_train_size = 'N/A'
                avg_val_size = 'N/A'
            
            analysis_results['metadata'] = {
                'refusal_class_names': CLASS_NAMES,
                'test_df': test_df,
                'predictions': [],
                'true_labels': [],
                'confidences': [],
                'cv_results': refusal_cv_results,
                'train_samples': avg_train_size,
                'val_samples': avg_val_size,
                'test_samples': test_size
            }

        # ═══════════════════════════════════════════════════════════
        # LOAD PRE-COMPUTED ANALYSIS RESULTS
        # ═══════════════════════════════════════════════════════════
        # WHY: These analyses are already done in RefusalPipeline.run_analyses()
        # when running full pipeline or resuming from step 9. We just load them here.
        
        print("\n--- Loading Pre-Computed Analysis Results ---")
        
        # Load power law analysis
        try:
            power_law_path = os.path.join(analysis_results_path, f"refusal_power_law_analysis_{model_timestamp}.pkl")
            if os.path.exists(power_law_path):
                with open(power_law_path, 'rb') as f:
                    analysis_results['power_law'] = pickle.load(f)
                print(f"✓ Loaded power law analysis from {power_law_path}")
            else:
                print(f"⚠️  Power law analysis not found at {power_law_path}")
                analysis_results['power_law'] = None
        except Exception as e:
            print(f"⚠️  Failed to load power law analysis: {e}")
            analysis_results['power_law'] = None
        
        # Load adversarial testing results
        try:
            adv_results_path = os.path.join(analysis_results_path, "adversarial_testing.json")
            if os.path.exists(adv_results_path):
                with open(adv_results_path, 'r') as f:
                    analysis_results['adversarial'] = json.load(f)
                print(f"✓ Loaded adversarial results from {adv_results_path}")
            else:
                print(f"⚠️  Adversarial results not found")
                analysis_results['adversarial'] = None
        except Exception as e:
            print(f"⚠️  Failed to load adversarial results: {e}")
            analysis_results['adversarial'] = None
        
        # Load jailbreak analysis
        try:
            # Try timestamped file first (correct structure)
            jailbreak_path = os.path.join(analysis_results_path, f"jailbreak_analysis_{model_timestamp}.json")
            if os.path.exists(jailbreak_path):
                with open(jailbreak_path, 'r') as f:
                    analysis_results['jailbreak'] = json.load(f)
                print(f"✓ Loaded jailbreak analysis from {jailbreak_path}")
            else:
                # Fallback to non-timestamped file
                jailbreak_path = os.path.join(analysis_results_path, "jailbreak_analysis.json")
                if os.path.exists(jailbreak_path):
                    with open(jailbreak_path, 'r') as f:
                        analysis_results['jailbreak'] = json.load(f)
                    print(f"✓ Loaded jailbreak analysis from {jailbreak_path}")
                else:
                    print(f"⚠️  Jailbreak analysis not found")
                    analysis_results['jailbreak'] = None
        except Exception as e:
            print(f"⚠️  Failed to load jailbreak analysis: {e}")
            analysis_results['jailbreak'] = None
        
        # Load jailbreak power law analysis
        try:
            jb_power_law_path = os.path.join(analysis_results_path, f"jailbreak_power_law_analysis_{model_timestamp}.pkl")
            if os.path.exists(jb_power_law_path):
                with open(jb_power_law_path, 'rb') as f:
                    analysis_results['jailbreak_power_law'] = pickle.load(f)
                print(f"✓ Loaded jailbreak power law analysis from {jb_power_law_path}")
            else:
                print(f"⚠️  Jailbreak power law analysis not found")
                analysis_results['jailbreak_power_law'] = None
        except Exception as e:
            print(f"⚠️  Failed to load jailbreak power law analysis: {e}")
            analysis_results['jailbreak_power_law'] = None
        
        # Load attention analysis
        try:
            attention_path = os.path.join(attention_analysis_path, f"refusal_attention_analysis_results_{model_timestamp}.json")
            if os.path.exists(attention_path):
                with open(attention_path, 'r') as f:
                    analysis_results['attention'] = json.load(f)
                print(f"✓ Loaded attention analysis from {attention_path}")
            else:
                print(f"⚠️  Attention analysis not found")
                analysis_results['attention'] = {}
        except Exception as e:
            print(f"⚠️  Failed to load attention analysis: {e}")
            analysis_results['attention'] = {}
        
        # Load SHAP analysis
        try:
            shap_path = os.path.join(shap_analysis_path, f"refusal_shap_analysis_results_{model_timestamp}.json")
            
            if os.path.exists(shap_path):
                with open(shap_path, 'r') as f:
                    analysis_results['shap'] = json.load(f)
                print(f"✓ Loaded SHAP analysis from {shap_path}")
            else:
                print(f"⚠️  SHAP analysis not found")
                analysis_results['shap'] = {}
        except Exception as e:
            print(f"⚠️  Failed to load SHAP analysis: {e}")
            analysis_results['shap'] = {}
        
        # Load correlation analysis
        try:
            correlation_path = os.path.join(analysis_results_path, "refusal_jailbreak_correlation.pkl")
            if os.path.exists(correlation_path):
                with open(correlation_path, 'rb') as f:
                    analysis_results['correlation'] = pickle.load(f)
                print(f"✓ Loaded correlation analysis from {correlation_path}")
            else:
                print(f"⚠️  Correlation analysis not found")
                analysis_results['correlation'] = {}
        except Exception as e:
            print(f"⚠️  Failed to load correlation analysis: {e}")
            analysis_results['correlation'] = {}
        
        # If jailbreak results weren't loaded, create minimal structure for visualizations
        if analysis_results['jailbreak'] is None:
            print("\n⚠️  Creating minimal jailbreak results structure for visualizations")
            analysis_results['jailbreak'] = {
                'predictions': {
                    'preds': [],
                    'labels': [],
                    'confidences': []
                }
            }

        # Generate visualizations
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)

        visualizer = Visualizer()
        jailbreak_visualizer = Visualizer(class_names=JAILBREAK_CLASS_NAMES)

        # ═══════════════════════════════════════════════════════════
        # LOAD TRAINING HISTORIES FROM CHECKPOINTS
        # ═══════════════════════════════════════════════════════════
        
        print("\n📈 Loading training histories from model checkpoints...")
        
        # Load refusal training history
        refusal_history = {}
        if 'history' in refusal_checkpoint:
            refusal_history = refusal_checkpoint['history']
            print(f"✓ Loaded refusal training history ({len(refusal_history.get('train_loss', []))} epochs)")
        else:
            print("⚠️  No training history found in refusal checkpoint")
        
        # Load jailbreak training history
        jailbreak_history = {}
        if 'history' in jailbreak_checkpoint:
            jailbreak_history = jailbreak_checkpoint['history']
            print(f"✓ Loaded jailbreak training history ({len(jailbreak_history.get('train_loss', []))} epochs)")
        else:
            print("⚠️  No training history found in jailbreak checkpoint")

        # ═══════════════════════════════════════════════════════════
        # TRAINING CURVES (BOTH CLASSIFIERS)
        # ═══════════════════════════════════════════════════════════
        
        print("\n📊 Generating training curves...")
        
        # Training curves - Refusal Classifier
        if refusal_history:
            visualizer.plot_training_curves(
                refusal_history,
                os.path.join(reporting_viz_path, f"refusal_training_curves_{model_timestamp}.png")
            )
            print("✓ Refusal training curves created")
        else:
            print("⚠️  Skipping refusal training curves - no history available")
        
        # Training curves - Jailbreak Detector
        if jailbreak_history:
            jailbreak_visualizer.plot_training_curves(
                jailbreak_history,
                os.path.join(reporting_viz_path, f"jailbreak_training_curves_{model_timestamp}.png")
            )
            print("✓ Jailbreak training curves created")
        else:
            print("⚠️  Skipping jailbreak training curves - no history available")

        # ═══════════════════════════════════════════════════════════
        # REFUSAL CLASSIFIER VISUALIZATIONS
        # ═══════════════════════════════════════════════════════════
        
        print("\n📊 Generating Refusal Classifier visualizations...")
        
        # Confusion matrix - Refusal
        cm = confusion_matrix(labels, preds)
        visualizer.plot_confusion_matrix(
            cm,
            os.path.join(reporting_viz_path, f"refusal_confusion_matrix_{model_timestamp}.png")
        )
        
        # Class distribution - Refusal
        visualizer.plot_class_distribution(
            labels,
            os.path.join(reporting_viz_path, f"refusal_class_distribution_{model_timestamp}.png")
        )

        # Per-class F1 - Refusal
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
            os.path.join(reporting_viz_path, f"refusal_per_class_f1_{model_timestamp}.png")
        )

        # Per-model F1 - Refusal
        per_model_f1 = {}
        if 'models' in analysis_results['per_model']:
            for model_name, model_data in analysis_results['per_model']['models'].items():
                if 'weighted_f1' in model_data:
                    per_model_f1[model_name] = model_data['weighted_f1']

        visualizer.plot_per_model_f1(
            per_model_f1,
            os.path.join(reporting_viz_path, f"refusal_per_model_f1_{model_timestamp}.png")
        )

        # Adversarial robustness - Refusal
        if analysis_results['adversarial']:
            visualizer.plot_adversarial_robustness(
                analysis_results['adversarial'],
                os.path.join(reporting_viz_path, f"refusal_adversarial_robustness_{model_timestamp}.png")
            )

        # Confidence distributions - Refusal
        if 'confidence_by_class' in analysis_results['confidence']:
            visualizer.plot_confidence_distributions(
                analysis_results['confidence']['confidence_by_class'],
                os.path.join(reporting_viz_path, f"refusal_confidence_distributions_{model_timestamp}.png")
            )

        # ═══════════════════════════════════════════════════════════
        # JAILBREAK DETECTOR VISUALIZATIONS
        # ═══════════════════════════════════════════════════════════
        
        print("\n📊 Generating Jailbreak Detector visualizations...")
        
        # Get jailbreak predictions
        if analysis_results.get('jailbreak') and 'predictions' in analysis_results['jailbreak']:
            jb_preds = analysis_results['jailbreak']['predictions'].get('preds', [])
            jb_labels = analysis_results['jailbreak']['predictions'].get('labels', [])
            jb_confidences = analysis_results['jailbreak']['predictions'].get('confidences', [])
            
            if len(jb_preds) > 0 and len(jb_labels) > 0:
                # Confusion matrix - Jailbreak
                jb_cm = confusion_matrix(jb_labels, jb_preds, labels=[0, 1])
                jailbreak_visualizer.plot_confusion_matrix(
                    jb_cm,
                    os.path.join(reporting_viz_path, f"jailbreak_confusion_matrix_{model_timestamp}.png")
                )
                
                # Class distribution - Jailbreak
                jailbreak_visualizer.plot_class_distribution(
                    jb_labels,
                    os.path.join(reporting_viz_path, f"jailbreak_class_distribution_{model_timestamp}.png")
                )
                
                # Calculate jailbreak metrics for visualization
                jb_report = classification_report(
                    jb_labels,
                    jb_preds,
                    labels=[0, 1],
                    target_names=JAILBREAK_CLASS_NAMES,
                    output_dict=True,
                    zero_division=0
                )
                
                # Per-class F1 - Jailbreak
                jb_per_class_f1 = {
                    JAILBREAK_CLASS_NAMES[0]: jb_report[JAILBREAK_CLASS_NAMES[0]]['f1-score'],
                    JAILBREAK_CLASS_NAMES[1]: jb_report[JAILBREAK_CLASS_NAMES[1]]['f1-score']
                }
                jailbreak_visualizer.plot_per_class_f1(
                    jb_per_class_f1,
                    os.path.join(reporting_viz_path, f"jailbreak_per_class_f1_{model_timestamp}.png")
                )
                
                # Confidence distribution - Jailbreak
                if len(jb_confidences) > 0:
                    jailbreak_visualizer.plot_confidence_distributions(
                        jb_labels,
                        jb_confidences,
                        os.path.join(reporting_viz_path, f"jailbreak_confidence_distributions_{model_timestamp}.png")
                    )
                
                print("✓ Jailbreak visualizations created")
            else:
                print("⚠️  No jailbreak predictions available for visualization")
        else:
            print("⚠️  No jailbreak results available for visualization")
            
            
        # ═══════════════════════════════════════════════════════
        # SAVING STRUCTURED ANALYSIS RESULTS
        # ═══════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("SAVING STRUCTURED ANALYSIS RESULTS")
        print("="*60)

        # Convert all analysis results to JSON-serializable format
        serializable_results = {
            'per_model': convert_to_serializable(analysis_results['per_model']),
            'confidence': convert_to_serializable(analysis_results['confidence']),
            'power_law': convert_to_serializable(analysis_results.get('power_law', {})),
            'adversarial': convert_to_serializable(analysis_results.get('adversarial', {})),
            'attention': convert_to_serializable(analysis_results.get('attention', {})),
            'shap': convert_to_serializable(analysis_results.get('shap', {})),
            'jailbreak': convert_to_serializable(analysis_results.get('jailbreak', {})),
            'jailbreak_power_law': convert_to_serializable(analysis_results.get('jailbreak_power_law', {})),
            'correlation': convert_to_serializable(analysis_results.get('correlation', {})),
            'predictions': {
                'preds': [int(p) for p in preds],
                'labels': [int(l) for l in labels],
                'confidences': [float(c) for c in confidences]
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_test_samples': len(test_df),
                'num_train_samples': self._get_split_size('train'),
                'num_val_samples': self._get_split_size('val'),
                'refusal_class_names': CLASS_NAMES,
                'jailbreak_class_names': JAILBREAK_CLASS_NAMES
            }
        }

        
        structured_results_path = os.path.join(analysis_results_path, f"refusal_jailbreak_analysis_results_complete_{model_timestamp}.json")

        with open(structured_results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        

        print(f"✓ Saved structured analysis results to {structured_results_path}")

        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE")
        print("="*60)
        
        return analysis_results
    
#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
