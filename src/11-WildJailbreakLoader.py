# WildJailbreakLoader Module 
#----------------------------
# Loads and samples from the WildJailbreak dataset for jailbreak detection supplementation.
# Dataset: https://huggingface.co/datasets/allenai/wildjailbreak
# 
# IMPROVEMENTS:
# - Full Config/Utils integration (NO HARDCODING!)
# - Handles new is_jailbreak_attempt field
# - Better error handling
# - Checkpoint support for large loads
# All imports are in 01-Imports.py
###############################################################################


class WildJailbreakLoader:
    """
    Loads successful jailbreak samples from AllenAI's WildJailbreak dataset.
    
    Purpose: Supplement training data when modern LLMs successfully defend against
    all jailbreak attempts in our pipeline, resulting in insufficient positive samples.
    
    Dataset: WildJailbreak contains 262K safety-training prompt-response pairs
    - adversarial_harmful: 82,728 successful jailbreak examples
    - adversarial_benign: 78,706 benign adversarial queries
    - vanilla_harmful: 50,050 direct harmful requests
    - vanilla_benign: 50,050 harmless prompts
    
    Citation:
        WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models
        Liwei Jiang et al., NeurIPS 2024
        https://arxiv.org/abs/2406.18510
    """
    
    def __init__(self, random_seed: int = None, checkpoint_manager: CheckpointManager = None):
        """
        Initialize WildJailbreak loader.
        
        Args:
            random_seed: Random seed for reproducible sampling (default: from WILDJAILBREAK_CONFIG)
            checkpoint_manager: Optional checkpoint manager for large loads
        """
        # Use config values - NO HARDCODING!
        self.random_seed = random_seed or WILDJAILBREAK_CONFIG['random_seed']
        self.dataset = None
        self.filtered_data = None
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            operation_name='wildjailbreak_loading'
        )
        
        # Statistics tracking
        self.stats = {
            'total_loaded': 0,
            'filtered_count': 0,
            'sampled_count': 0,
            'loading_time': 0,
            'unique_tactics': 0
        }
        
        # Verbosity from config
        self.verbose = EXPERIMENT_CONFIG.get('verbose', True)

    def load_dataset(self) -> bool:
        """
        Load WildJailbreak dataset from HuggingFace.
        
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            if self.verbose:
                print_banner("LOADING WILDJAILBREAK DATASET", char="=")
                print(f"  Source: {WILDJAILBREAK_CONFIG['dataset_name']}")
                print(f"  Config: {WILDJAILBREAK_CONFIG['dataset_config']}")
                print(f"  Loading with TSV parameters...")
            
            # Check for datasets library
            try:
                load_dataset  # Already imported from 01-Imports.py
            except NameError:
                print("❌ ERROR: 'datasets' library not installed")
                print("   Install with: pip install datasets")
                return False
            
            # Load dataset from config
            try:
                # WildJailbreak requires specific parameters
                self.dataset = load_dataset(
                    WILDJAILBREAK_CONFIG['dataset_name'],
                    WILDJAILBREAK_CONFIG['dataset_config'],
                    delimiter="\t",           # Required: TSV format
                    keep_default_na=False,    # Required: Preserve data integrity
                    token=True                # Use HuggingFace token if available
                )
                
                # Get dataset size
                if hasattr(self.dataset, '__len__'):
                    self.stats['total_loaded'] = len(self.dataset)
                elif 'train' in self.dataset:
                    self.stats['total_loaded'] = len(self.dataset['train'])
                
                if self.verbose:
                    print(f"  ✓ Loaded {self.stats['total_loaded']:,} total samples")
                
            except Exception as auth_error:
                # Check if gated dataset issue
                error_msg = str(auth_error).lower()
                if 'gated' in error_msg or 'authenticated' in error_msg:
                    self._print_authentication_guide()
                raise
            
            self.stats['loading_time'] = time.time() - start_time
            
            if self.verbose:
                print(f"  Loading time: {format_time(self.stats['loading_time'])}")
                print("=" * 60)
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ ERROR loading WildJailbreak: {str(e)[:200]}")
            return False

    def _print_authentication_guide(self):
        """Print helpful authentication instructions."""
        print_banner("WILDJAILBREAK AUTHENTICATION REQUIRED", char="!")
        print("\nWildJailbreak is a GATED dataset. You need to:")
        print("  1. Create HuggingFace account: https://huggingface.co/join")
        print("  2. Request access: https://huggingface.co/datasets/allenai/wildjailbreak")
        print("  3. Get token: https://huggingface.co/settings/tokens")
        print("  4. Login: huggingface-cli login")
        print("\nAfter authentication, re-run your experiment.")
        print("!" * 60)

    def filter_successful_jailbreaks(self) -> pd.DataFrame:
        """
        Filter for successful jailbreak samples (adversarial_harmful).
        
        Returns:
            DataFrame with filtered samples
        """
        if self.dataset is None:
            print("❌ ERROR: Dataset not loaded. Call load_dataset() first.")
            return pd.DataFrame()
        
        try:
            if self.verbose:
                print("\n🔍 Filtering for successful jailbreaks...")
            
            # Convert to pandas
            if hasattr(self.dataset, 'to_pandas'):
                df = self.dataset.to_pandas()
            else:
                # DatasetDict - access the split
                if 'train' in self.dataset:
                    df = self.dataset['train'].to_pandas()
                else:
                    first_split = list(self.dataset.keys())[0]
                    if self.verbose:
                        print(f"  Using split: {first_split}")
                    df = self.dataset[first_split].to_pandas()
            
            # Filter for specified data type from config
            data_type = WILDJAILBREAK_CONFIG['data_type_filter']
            self.filtered_data = df[df['data_type'] == data_type].copy()
            self.stats['filtered_count'] = len(self.filtered_data)
            
            if self.verbose:
                print(f"  ✓ Found {self.stats['filtered_count']:,} {data_type} samples")
                
                # Show field statistics
                if 'tactics' in self.filtered_data.columns:
                    tactics_expanded = self.filtered_data.explode('tactics')
                    unique_tactics = tactics_expanded['tactics'].nunique()
                    self.stats['unique_tactics'] = unique_tactics
                    print(f"  ✓ Found {unique_tactics} unique jailbreak tactics")
            
            return self.filtered_data
            
        except Exception as e:
            print(f"❌ ERROR filtering data: {str(e)[:200]}")
            return pd.DataFrame()

    def sample_diverse(self, n_samples: int) -> pd.DataFrame:
        """
        Sample diverse jailbreak examples with comprehensive validation.
        
        Args:
            n_samples: Number of samples to retrieve (must be positive integer)
        
        Returns:
            DataFrame with sampled examples in standardized format
            
        Raises:
            TypeError: If n_samples is not an integer
            ValueError: If n_samples is invalid or insufficient data available
        """
        # Input validation - type check
        if not isinstance(n_samples, int):
            raise TypeError(f"n_samples must be int, got {type(n_samples).__name__}")
        
        # Input validation - range check
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        
        # Data availability check
        if self.filtered_data is None:
            raise ValueError(
                "No filtered data available. Call filter_successful_jailbreaks() first."
            )
        
        if len(self.filtered_data) == 0:
            raise ValueError("Filtered dataset is empty. No samples to return.")
        
        try:
            # Adjust if requesting more than available
            available = len(self.filtered_data)
            if n_samples > available:
                if self.verbose:
                    print(f"⚠️  Requested {n_samples:,} samples but only {available:,} available")
                    print(f"   Sampling all {available:,} samples")
                n_samples = available
            
            if self.verbose:
                print_banner(f"SAMPLING {n_samples:,} JAILBREAK EXAMPLES", char="─")
            
            # Set random seed for reproducibility
            np.random.seed(self.random_seed)
            
            # Check for diversity fields from config
            diversity_fields = WILDJAILBREAK_CONFIG.get('diversity_fields', [])
            
            if diversity_fields and any(field in self.filtered_data.columns for field in diversity_fields):
                # Diversity sampling based on tactics
                if 'tactics' in diversity_fields and 'tactics' in self.filtered_data.columns:
                    sampled = self._sample_by_tactics(n_samples)
                else:
                    # Random sampling
                    sampled = self.filtered_data.sample(n=n_samples, random_state=self.random_seed)
            else:
                # Random sampling if no diversity fields
                if self.verbose:
                    print("  Using random sampling")
                sampled = self.filtered_data.sample(n=n_samples, random_state=self.random_seed)
            
            self.stats['sampled_count'] = len(sampled)
            
            if self.verbose:
                print(f"  ✓ Sampled {self.stats['sampled_count']:,} examples")
            
            # Convert to standardized format
            standardized = self._standardize_format(sampled)
            
            # Output validation - ensure required columns present
            required_cols = ['prompt', 'response', 'model']
            missing_cols = [col for col in required_cols if col not in standardized.columns]
            if missing_cols:
                raise ValueError(f"Standardized data missing required columns: {missing_cols}")
            
            # Output validation - ensure not empty
            if len(standardized) == 0:
                raise ValueError("Sampling produced empty dataset after standardization")
            
            # Output validation - check for null values in critical columns
            null_counts = standardized[required_cols].isnull().sum()
            if null_counts.any():
                raise ValueError(
                    f"Standardized data contains null values in required columns: "
                    f"{null_counts[null_counts > 0].to_dict()}"
                )
            
            return standardized
            
        except (TypeError, ValueError):
            # Re-raise validation errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise RuntimeError(f"Error sampling data: {str(e)[:200]}") from e

    def _sample_by_tactics(self, n_samples: int) -> pd.DataFrame:
        """Sample proportionally across different jailbreak tactics."""
        # Explode tactics to get counts
        tactics_expanded = self.filtered_data.explode('tactics')
        tactic_counts = tactics_expanded['tactics'].value_counts()
        
        if self.verbose:
            print(f"  Sampling across {len(tactic_counts)} tactics")
            print("  Top 5 tactics:")
            for tactic, count in tactic_counts.head(5).items():
                print(f"    - {tactic}: {count:,} samples")
        
        # Sample proportionally
        sampled_indices = []
        samples_per_tactic = max(1, n_samples // len(tactic_counts))
        
        for tactic in tactic_counts.index:
            # Get samples with this tactic
            tactic_mask = self.filtered_data['tactics'].apply(
                lambda x: tactic in x if isinstance(x, list) else False
            )
            tactic_indices = self.filtered_data[tactic_mask].index.tolist()
            
            # Sample from this tactic
            n_to_sample = min(samples_per_tactic, len(tactic_indices))
            if n_to_sample > 0:
                sampled_indices.extend(
                    np.random.choice(tactic_indices, n_to_sample, replace=False)
                )
        
        # Fill remaining with random samples
        if len(sampled_indices) < n_samples:
            remaining_indices = list(set(self.filtered_data.index) - set(sampled_indices))
            additional_needed = n_samples - len(sampled_indices)
            if remaining_indices:
                sampled_indices.extend(
                    np.random.choice(
                        remaining_indices,
                        min(additional_needed, len(remaining_indices)),
                        replace=False
                    )
                )
        
        return self.filtered_data.loc[sampled_indices[:n_samples]].copy()

    def _standardize_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert WildJailbreak format to pipeline's standardized format.

        Expected format:
        - prompt: The jailbreak prompt text
        - response: The model response
        - refusal_label: Classification (from config, not hardcoded!)
        - is_jailbreak_attempt: 1 (these are all jailbreak attempts)
        - jailbreak_label: 1 (these all succeeded)
        - confidence scores: From config
        """
        try:
            standardized = pd.DataFrame()
            
            # Map prompt field
            if 'adversarial' in df.columns and 'vanilla' in df.columns:
                # Use adversarial if non-empty, otherwise vanilla
                standardized['prompt'] = df.apply(
                    lambda row: row['adversarial'] if (pd.notna(row['adversarial']) and str(row['adversarial']).strip())
                               else row['vanilla'] if pd.notna(row['vanilla']) else '',
                    axis=1
                )
            elif 'adversarial' in df.columns:
                standardized['prompt'] = df['adversarial']
            elif 'vanilla' in df.columns:
                standardized['prompt'] = df['vanilla']
            elif 'prompt' in df.columns:
                standardized['prompt'] = df['prompt']
            else:
                print("❌ ERROR: No prompt field found")
                return pd.DataFrame()
            
            # Map response field
            if 'completion' in df.columns:
                standardized['response'] = df['completion'].fillna('')
            elif 'response' in df.columns:
                standardized['response'] = df['response'].fillna('')
            else:
                print("❌ ERROR: No response field found")
                return pd.DataFrame()
            
            # Labels from config - NO HARDCODING!
            standardized['expected_label'] = 'jailbreaks'
            
            # Refusal label - use config value
            standardized['refusal_label'] = WILDJAILBREAK_CONFIG.get('default_refusal_label', 0)
            
            # NEW: Set jailbreak attempt field
            standardized['is_jailbreak_attempt'] = 1  # These are all jailbreak attempts

            # Jailbreak label (all succeeded in WildJailbreak)
            # All WildJailbreak samples are successful jailbreaks
            standardized['jailbreak_label'] = 1  # Changed from jailbreak_success to match pipeline expectation
            
            # Confidence scores from config - NO HARDCODING!
            standardized['refusal_confidence'] = WILDJAILBREAK_CONFIG.get('default_confidence', 90)
            standardized['jailbreak_confidence'] = WILDJAILBREAK_CONFIG.get('default_confidence', 90)
            
            # Metadata
            standardized['data_source'] = 'wildjailbreak'
            standardized['model'] = WILDJAILBREAK_CONFIG['synthetic_model_label']
            standardized['timestamp'] = get_timestamp('display')
            
            # Optional: preserve tactics
            if 'tactics' in df.columns:
                standardized['wildjailbreak_tactics'] = df['tactics']
            
            if self.verbose:
                print(f"  ✓ Standardized {len(standardized):,} samples to pipeline format")
            
            return standardized
            
        except Exception as e:
            print(f"❌ ERROR standardizing format: {str(e)[:200]}")
            return pd.DataFrame()

    def load_and_sample(self, n_samples: int, save_checkpoint: bool = False) -> pd.DataFrame:
        """
        Convenience method: Load dataset, filter, and sample in one call.
        
        Args:
            n_samples: Number of samples to retrieve
            save_checkpoint: Whether to save checkpoint after loading
        
        Returns:
            DataFrame with sampled examples in standardized format
        """
        # Try to load from checkpoint first
        if save_checkpoint:
            checkpoint = self.checkpoint_manager.load_latest_checkpoint()
            if checkpoint:
                if self.verbose:
                    print("✅ Loaded WildJailbreak data from checkpoint")
                return checkpoint['data']
        
        # Load dataset
        if not self.load_dataset():
            return pd.DataFrame()
        
        # Filter for successful jailbreaks
        filtered = self.filter_successful_jailbreaks()
        if len(filtered) == 0:
            return pd.DataFrame()
        
        # Sample diversely
        sampled = self.sample_diverse(n_samples)
        
        # Save checkpoint if requested
        if save_checkpoint and len(sampled) > 0:
            self.checkpoint_manager.save_checkpoint(
                data=sampled,
                last_index=len(sampled),
                metadata={'stats': self.stats}
            )
            if self.verbose:
                print(f"💾 Saved checkpoint with {len(sampled):,} samples")
        
        return sampled

    def get_statistics(self) -> Dict:
        """Get loader statistics."""
        return {
            **self.stats,
            'random_seed': self.random_seed,
            'config': WILDJAILBREAK_CONFIG
        }


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
